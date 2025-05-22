import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 定义h_sigmoid激活函数，这是一种硬Sigmoid函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数


# 定义h_swish激活函数，这是基于h_sigmoid的Swish函数变体
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)


# 定义Coordinate Attention模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 定义水平和垂直方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = max(8, inp // reduction)  # 计算中间层的通道数

        # 1x1卷积用于降维
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 激活函数

        # 两个1x1卷积，分别对应水平和垂直方向
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 保存输入作为残差连接

        n, c, h, w = x.size()  # 获取输入的尺寸
        x_h = self.pool_h(x)  # 水平方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 垂直方向池化并交换维度以适应拼接

        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的特征
        y = self.conv1(y)  # 通过1x1卷积降维
        y = self.bn1(y)  # 批归一化
        y = self.act(y)  # 激活函数

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 将特征拆分回水平和垂直方向
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复x_w的原始维度

        a_h = self.conv_h(x_h).sigmoid()  # 通过1x1卷积并应用Sigmoid获取水平方向的注意力权重
        a_w = self.conv_w(x_w).sigmoid()  # 通过1x1卷积并应用Sigmoid获取垂直方向的注意力权重

        out = identity * a_w * a_h  # 应用注意力权重到输入特征，并与残差连接相乘

        return out  # 返回输出


# GroupBatchnorm2d模块是对标准批量归一化的扩展，它将特征通道分组进行归一化。
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num, "通道数必须大于等于分组数"
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))  # 偏置参数
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.group_num, -1)  # [N, G, C//G * H * W]
        mean = x.mean(dim=2, keepdim=True)  # [N, G, 1]
        std = x.std(dim=2, keepdim=True)  # [N, G, 1]
        x = (x - mean) / (std + self.eps)  # 标准化
        x = x.view(N, C, H, W)  # 恢复形状
        return x * self.weight + self.bias  # 应用权重和偏置


# SRU模块用于抑制空间冗余。它通过分组归一化和一个门控机制实现。
class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = False):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)  # 归一化权重
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigmoid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


# CRU模块用于处理通道冗余。它通过一个压缩-卷积-扩展策略来增强特征的代表性。
class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super(CRU, self).__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


# ScConv模块结合了SRU和CRU两个子模块，用于同时处理空间和通道冗余。
class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 4, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super(ScConv, self).__init__()
        self.SRU = SRU(oup_channels=op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel=op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


# 定义 KmerDatasetWithEmbeddingAndLabel 类
class KmerDatasetWithEmbeddingAndLabel(Dataset):
    def __init__(self, saved_data=None):
        if saved_data is not None:
            self.feature_representations = saved_data['features']
            self.embeddings = saved_data['embeddings']
            self.sequence_ids = saved_data['sequence_ids']
            self.labels = saved_data['labels']
            self.label_mapping = saved_data['label_mapping']
            print("已从保存的文件加载数据。")
        else:
            raise ValueError("必须提供保存的数据。")

    def filter_data(self):
        # 记录过滤前的样本数量
        original_len = len(self.labels)

        # 获取标签不为2的索引
        valid_indices = [i for i, label in enumerate(self.labels) if label != 2]

        # 过滤数据，去除label为2的样本
        self.feature_representations = [self.feature_representations[i] for i in valid_indices]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.sequence_ids = [self.sequence_ids[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

        # 计算并打印移除的样本数量
        removed = original_len - len(valid_indices)
        print(f"已移除 {removed} 个标签为2的样本。")

    def __len__(self):
        return len(self.feature_representations)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature_representations[idx], dtype=torch.float32)  # [64, 64, 2]
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)  # [768]
        label = self.labels[idx]
        seq_id = self.sequence_ids[idx]
        return feature.permute(2, 0, 1), embedding, label, seq_id  # 调整feature的维度以匹配卷积层的输入


# 修改 CustomDataLoader，使其接受文件列表并处理所有批次
class CustomDataLoader:
    def __init__(self, data_path, files=None, batch_size=50, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        if files is None:
            self.files = sorted(
                [f for f in os.listdir(data_path) if f.endswith('.pt')],
                key=lambda x: int(re.search(r'\d+', x).group())
            )
        else:
            self.files = files

        self.num_batches = math.ceil(len(self.files) / self.batch_size)  # 使用math.ceil
        self.current_batch = 0

        if self.shuffle:
            np.random.shuffle(self.files)

    def __iter__(self):
        self.current_batch = 0  # 重置批次计数器
        if self.shuffle:
            np.random.shuffle(self.files)  # 每次迭代重新打乱
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_files = self.files[start_idx:end_idx]
        batch_data = []

        # 打印当前加载的批次文件信息
        print("\n-------------------------------------------------------")
        print(f"正在加载第 {self.current_batch + 1}/{self.num_batches} 批次的文件: {batch_files}")

        for file in batch_files:
            file_path = os.path.join(self.data_path, file)
            data = torch.load(file_path)
            batch_data.append(data)

        combined_data = {
            'features': np.concatenate([d['features'] for d in batch_data], axis=0),
            'embeddings': np.concatenate([d['embeddings'] for d in batch_data], axis=0),
            'sequence_ids': np.concatenate([d['sequence_ids'] for d in batch_data], axis=0),
            'labels': np.concatenate([d['labels'] for d in batch_data], axis=0),
            'label_mapping': batch_data[0]['label_mapping']
        }

        dataset = KmerDatasetWithEmbeddingAndLabel(saved_data=combined_data)
        dataset.filter_data()

        self.current_batch += 1
        return dataset


# 定义早停机制
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        :param patience: 如果验证准确度在连续的patience个epoch中没有提升，则提前停止训练。
        :param delta: 如果验证准确度的变化小于delta，则不认为是改善。
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_acc = None

    def step(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
            return False
        elif val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0  # 重置计数器
            return False
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("触发早停机制！")
            return True
        return False


# 定义深度学习模型
class VirusClassifier(nn.Module):
    def __init__(self):
        super(VirusClassifier, self).__init__()

        # 特征矩阵分支
        # 替换标准卷积层为ScConv模块
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.scconv1 = ScConv(op_channel=32, group_num=4, gate_treshold=0.5, alpha=1 / 2, squeeze_radio=2, group_size=2,
                              group_kernel_size=3)
        self.ca1 = CoordAtt(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.scconv2 = ScConv(op_channel=64, group_num=4, gate_treshold=0.5, alpha=1 / 2, squeeze_radio=2, group_size=2,
                              group_kernel_size=3)
        self.ca2 = CoordAtt(64, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.scconv3 = ScConv(op_channel=128, group_num=4, gate_treshold=0.5, alpha=1 / 2, squeeze_radio=2,
                              group_size=2, group_kernel_size=3)
        self.ca3 = CoordAtt(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(0.5)

        # 嵌入向量分支
        self.fc_embed1 = nn.Linear(768, 512)
        self.attention_embed = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.dropout_embed1 = nn.Dropout(0.5)
        self.fc_embed2 = nn.Linear(512, 256)
        self.dropout_embed2 = nn.Dropout(0.5)

        # 融合后全连接层
        self.fc_combined1 = nn.Linear(256 + 256, 256)
        self.dropout_combined1 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, 1)  # 二分类

    def forward(self, feature_matrix, embedding_vector):
        # 特征矩阵分支
        x = F.relu(self.conv1(feature_matrix))  # 原始卷积层
        x = self.scconv1(x)  # ScConv模块
        x = self.ca1(x)  # Coordinate Attention
        x = self.pool1(x)  # 池化

        x = F.relu(self.conv2(x))
        x = self.scconv2(x)
        x = self.ca2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.scconv3(x)
        x = self.ca3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 嵌入向量分支
        y = F.relu(self.fc_embed1(embedding_vector))
        y = y.unsqueeze(0)  # 添加序列长度维度，形状变为 [1, batch_size, 512]
        y, _ = self.attention_embed(y, y, y)  # 自注意力
        y = y.squeeze(0)  # 移除序列长度维度，形状恢复为 [batch_size, 512]
        y = self.dropout_embed1(y)
        y = F.relu(self.fc_embed2(y))
        y = self.dropout_embed2(y)

        # 融合
        combined = torch.cat((x, y), dim=1)
        combined = F.relu(self.fc_combined1(combined))
        combined = self.dropout_combined1(combined)
        out = self.fc_out(combined)  # 移除 sigmoid

        return out

# 定义辅助函数用于解析文件名和创建复合标签
def parse_filename(filename):
    """
    从文件名中提取k-mer长度和类别。
    例如，'100-400_Eukaryotes Virus1.pt' 会提取到：
    kmer_length: '100-400bp'
    category: 'Eukaryotes_Virus'
    """
    basename = os.path.basename(filename)
    pattern = r'(\d+-\d+)_([A-Za-z ]+)\d+\.pt'
    match = re.match(pattern, basename)
    if match:
        kmer_length = f"{match.group(1)}bp"
        category = match.group(2).strip().replace(" ", "_")  # 替换空格为下划线以避免问题
        return f"{category}_{kmer_length}"
    else:
        return "Unknown_Unknown"
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    batch_number = 0  # 外部批次计数器

    for batch in train_loader:
        batch_number += 1
        batch_loss = 0
        batch_correct = 0
        batch_total = 0

        # 使用内层 DataLoader 处理每个外部批次的样本
        data_loader = DataLoader(batch, batch_size=32, shuffle=True)
        for features, embeddings, labels, _ in data_loader:
            features = features.to(device)
            embeddings = embeddings.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(features, embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() * features.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            batch_correct += (preds == labels.byte()).sum().item()
            batch_total += labels.size(0)
            torch.cuda.empty_cache()  # 清理CUDA内存

        # 计算并打印外部批次的平均损失和准确率
        batch_loss_avg = batch_loss / batch_total
        batch_acc = batch_correct / batch_total
        print(
            f"外部批次 {batch_number}/{len(train_loader)} - 训练损失: {batch_loss_avg:.4f}, 训练准确率: {batch_acc:.4f}")

        # 累积到整个epoch的指标
        epoch_loss += batch_loss
        epoch_correct += batch_correct
        epoch_total += batch_total

    # 计算整个epoch的平均损失和准确率
    epoch_loss_avg = epoch_loss / epoch_total
    epoch_acc = epoch_correct / epoch_total
    return epoch_loss_avg, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds, all_targets, all_probs = [], [], []  # 初始化列表

    batch_number = 0  # 外部批次计数器

    with torch.no_grad():
        for batch in val_loader:
            batch_number += 1
            batch_loss = 0
            batch_correct = 0
            batch_total = 0

            # 使用内层 DataLoader 处理每个外部批次的样本
            data_loader = DataLoader(batch, batch_size=32, shuffle=False)
            for features, embeddings, labels, _ in data_loader:
                features = features.to(device)
                embeddings = embeddings.to(device)
                labels = labels.float().unsqueeze(1).to(device)  # [batch_size, 1]

                outputs = model(features, embeddings)
                loss = criterion(outputs, labels)
                batch_loss += loss.item() * features.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                batch_correct += (preds == labels).sum().item()
                batch_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                torch.cuda.empty_cache()  # 清理CUDA内存

            # 计算并打印外部批次的平均损失和准确率
            batch_loss_avg = batch_loss / batch_total
            batch_acc = batch_correct / batch_total
            print(
                f"外部批次 {batch_number}/{len(val_loader)} - 验证损失: {batch_loss_avg:.4f}, 验证准确率: {batch_acc:.4f}")

            # 累积到整个验证集的指标
            val_loss += batch_loss
            val_correct += batch_correct
            val_total += batch_total

    # 计算整体验证指标
    accuracy = val_correct / val_total
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0

    # 计算整体平均损失
    overall_val_loss = val_loss / val_total

    return overall_val_loss, accuracy, f1, roc_auc

# 设置随机种子以确保可复现性，并设置CUDA的确定性行为
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. 数据和路径设置
data_path = r"/home/zhangjianpeng/Virus_identity/data/pytorch_dataset_1200-1800"
model_save_dir = r"/home/zhangjianpeng/Virus_identity/model"
os.makedirs(model_save_dir, exist_ok=True)

# 获取所有.pt文件，并按数字顺序排序
all_files = sorted(
    [f for f in os.listdir(data_path) if f.endswith('.pt')],
    key=lambda x: int(re.search(r'\d+', x).group())
)

# 创建复合标签列表
composite_labels = [parse_filename(f) for f in all_files]

# 使用LabelEncoder将复合标签转换为整数
le = LabelEncoder()
composite_labels_encoded = le.fit_transform(composite_labels)
# 定义StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(skf.split(all_files, composite_labels_encoded))

# 3. 定义训练和验证函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. 五折交叉验证训练和验证过程
results = []

num_epochs = 20  # 设置训练epoch的数量
patience = 5
delta = 0.001
learning_rate = 0.001

for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"\nFold {fold_idx + 1}/5")
    print("-" * 50)

    # 获取训练和验证文件
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    # 加载数据
    train_loader = CustomDataLoader(data_path, files=train_files, batch_size=20, shuffle=True)
    val_loader = CustomDataLoader(data_path, files=val_files, batch_size=20, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = VirusClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  # 移除 verbose=True

    # 初始化每个fold的早停机制
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    # 跟踪当前fold的最佳验证准确率和模型权重
    best_val_acc_fold = 0.0
    best_model_wts_fold = None

    # 开始训练
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_roc_auc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val ROC AUC: {val_roc_auc:.4f}")

        # 更新学习率调度器
        scheduler.step(val_loss)

        # # 获取并打印当前学习率
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"当前学习率: {current_lr:.6f}")

        # 早停机制
        if early_stopping.step(val_acc):
            print("触发fold早停机制！")
            break

        # 检查验证性能是否改善，更新fold的最佳模型
        if val_acc > best_val_acc_fold:
            best_val_acc_fold = val_acc
            best_model_wts_fold = model.state_dict()
            torch.save(best_model_wts_fold, os.path.join(model_save_dir, f"best_model_fold_{fold_idx + 1}.pth"))
            print(f"当前fold保存新最佳模型，验证准确率: {best_val_acc_fold:.4f}")

    # 保存本fold的结果
    results.append({
        'fold': fold_idx + 1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_roc_auc': val_roc_auc
    })

# 5. 汇总结果
results_df = pd.DataFrame(results)
print("\n五折交叉验证结果:")
print(results_df)

print("\n平均性能:")
print(results_df.mean())

# 6. 保存全局最佳模型（此方案不再使用全局最佳模型）

# # 7. 方案B：训练一个最终模型
# print("\n开始训练最终模型...")
#
# # 获取所有数据并划分为80%训练集和20%验证集
# final_train_files, final_val_files = train_test_split(all_files, test_size=0.2, random_state=42, shuffle=True)
#
# # 加载数据
# final_train_loader = CustomDataLoader(data_path, files=final_train_files, batch_size=20, shuffle=True)
# final_val_loader = CustomDataLoader(data_path, files=final_val_files, batch_size=20, shuffle=False)
#
# # 初始化模型、损失函数和优化器
# final_model = VirusClassifier().to(device)
# final_criterion = nn.BCEWithLogitsLoss()
# final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
# final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.1, patience=3)  # 移除 verbose=True
#
# # 初始化早停机制
# final_early_stopping = EarlyStopping(patience=patience, delta=delta)
#
# # 跟踪最终模型的最佳验证准确率和模型权重
# best_val_acc_final = 0.0
# best_model_wts_final = None
#
# # 训练最终模型
# for epoch in range(num_epochs):
#     print(f"Final Model - Epoch {epoch + 1}")
#     train_loss, train_acc = train_one_epoch(final_model, final_train_loader, final_criterion, final_optimizer, device)
#     val_loss, val_acc, val_f1, val_roc_auc = validate(final_model, final_val_loader, final_criterion, device)
#
#     print(f"Final Train Loss: {train_loss:.4f}, Final Train Acc: {train_acc:.4f}")
#     print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}, Final Val F1: {val_f1:.4f}, Final Val ROC AUC: {val_roc_auc:.4f}")
#
#     # 更新学习率调度器
#     final_scheduler.step(val_loss)
#
#     # 获取并打印当前学习率
#     final_current_lr = final_scheduler.get_last_lr()[0]
#     print(f"当前学习率: {final_current_lr:.6f}")
#
#     # 早停机制
#     if final_early_stopping.step(val_acc):
#         print("触发最终模型早停机制！")
#         break
#
#     # 检查验证性能是否改善，保存最终模型的最佳模型
#     if val_acc > best_val_acc_final:
#         best_val_acc_final = val_acc
#         best_model_wts_final = final_model.state_dict()
#         torch.save(best_model_wts_final, os.path.join(model_save_dir, "final_best_model.pth"))
#         print(f"最终模型保存新最佳模型，验证准确率: {best_val_acc_final:.4f}")
#
# print("\n最终模型训练结束。已保存最佳最终模型到: final_best_model.pth")
