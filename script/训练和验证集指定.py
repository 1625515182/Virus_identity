import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# 定义激活函数和注意力模块（保持不变）
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

# GroupBatchnorm2d和其他模块保持不变
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num, "通道数必须大于等于分组数"
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = False):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
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

# 定义综合样本级别的 Dataset
class CombinedDataset(Dataset):
    def __init__(self, data_path, files):
        self.data_path = data_path
        self.files = files
        self.features = []
        self.embeddings = []
        self.sequence_ids = []
        self.labels = []
        self.label_mapping = {}
        self._load_data()

    def _load_data(self):
        for file in self.files:
            file_path = os.path.join(self.data_path, file)
            try:
                data = torch.load(file_path)
                self.features.extend(data['features'])
                self.embeddings.extend(data['embeddings'])
                self.sequence_ids.extend(data['sequence_ids'])
                self.labels.extend(data['labels'])
                if 'label_mapping' in data:
                    self.label_mapping = data['label_mapping']
            except Exception as e:
                print(f"错误: 加载文件 {file} 时出错: {e}")

        # 过滤掉标签为2的样本
        original_len = len(self.labels)
        valid_indices = [i for i, label in enumerate(self.labels) if label != 2]
        self.features = [self.features[i] for i in valid_indices]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.sequence_ids = [self.sequence_ids[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        removed = original_len - len(valid_indices)
        print(f"已移除 {removed} 个标签为2的样本。")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32).permute(2, 0, 1)  # [2, 64, 64]
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)  # [768]
        label = self.labels[idx]
        seq_id = self.sequence_ids[idx]
        return feature, embedding, label, seq_id

# 定义早停机制（基于验证损失）
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        """
        :param patience: 如果验证损失在连续的patience个epoch中没有降低，则提前停止训练。
        :param delta: 验证损失的最低变化量，只有变化大于delta才认为是改善。
        :param path: 保存最佳模型的路径。
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.best_model_wts = None
        self.path = path

    def step(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0  # 重置计数器
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
        self.attention_embed = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.dropout_embed1 = nn.Dropout(0.5)
        self.fc_embed2 = nn.Linear(512, 256)
        self.dropout_embed2 = nn.Dropout(0.5)

        # 融合后全连接层
        self.fc_combined1 = nn.Linear(256 + 256, 256)
        self.dropout_combined1 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, 1)  # 二分类

    def forward(self, feature_matrix, embedding_vector):
        # 特征矩阵分支
        x = F.relu(self.conv1(feature_matrix))
        x = self.scconv1(x)
        x = self.ca1(x)
        x = self.pool1(x)

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
        y = y.unsqueeze(1)  # 添加序列长度维度，形状变为 [batch_size, 1, 512]
        y, _ = self.attention_embed(y, y, y)  # 自注意力
        y = y.squeeze(1)  # 移除序列长度维度，形状恢复为 [batch_size, 512]
        y = self.dropout_embed1(y)
        y = F.relu(self.fc_embed2(y))
        y = self.dropout_embed2(y)

        # 融合
        combined = torch.cat((x, y), dim=1)
        combined = F.relu(self.fc_combined1(combined))
        combined = self.dropout_combined1(combined)
        out = self.fc_out(combined)  # 移除 sigmoid

        return out

# 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 定义训练集和验证集的路径
    train_path = r"/home/zhangjianpeng/Virus_identity/data/pytorch_dataset_400-800"
    val_path = r"/home/zhangjianpeng/Virus_identity/data/pytorch_dataset_400-800_val"

    # 获取训练集文件列表
    train_files = sorted(
        [f for f in os.listdir(train_path) if f.endswith('.pt')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )

    # 获取验证集文件列表
    val_files = sorted(
        [f for f in os.listdir(val_path) if f.endswith('.pt')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )

    # 打印文件信息
    print(f"训练集文件数: {len(train_files)}")
    print(f"验证集文件数: {len(val_files)}")

    # 创建训练和验证的 CombinedDataset
    train_dataset = CombinedDataset(train_path, train_files)
    val_dataset = CombinedDataset(val_path, val_files)

    # 打印标签分布
    print("训练集标签分布:", Counter(train_dataset.labels))
    print("验证集标签分布:", Counter(val_dataset.labels))

    # 计算类权重以应对可能的类别不平衡
    label_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset)
    class_weights = {label: total_samples / count for label, count in label_counts.items()}
    weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 定义模型保存目录
    model_save_dir = r"/home/zhangjianpeng/Virus_identity/model"
    os.makedirs(model_save_dir, exist_ok=True)  # 如果目录不存在，则创建

    # 初始化模型、优化器、损失函数、学习率调度器、早停机制
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    model = VirusClassifier().to(device)

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    # 设置最佳模型保存路径
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=best_model_path)

    num_epochs = 50  # 增加训练轮数
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        model.train()
        epoch_running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # 训练循环
        for batch_idx, (features, embeddings, labels, seq_ids) in enumerate(train_loader):
            # 将数据移动到设备
            features = features.to(device)
            embeddings = embeddings.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(features, embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累加损失
            epoch_running_loss += loss.item() * features.size(0)  # 乘以batch_size以便后续平均

            # 计算预测结果
            preds = torch.sigmoid(outputs) >= 0.5
            epoch_correct += (preds.float() == labels).sum().item()
            epoch_total += labels.size(0)

            # 可选：打印训练进度
            # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                # print(f"  批次 {batch_idx + 1}/{len(train_loader)} - 当前损失: {loss.item():.4f}")

        # 计算并打印本epoch的平均训练损失和准确率
        epoch_loss = epoch_running_loss / epoch_total
        epoch_accuracy = epoch_correct / epoch_total
        print(f"\n本epoch训练平均损失: {epoch_loss:.4f} - 准确率: {epoch_accuracy:.4f}")

        # 验证循环
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for batch_idx, (features, embeddings, labels, seq_ids) in enumerate(val_loader):
                # 将数据移动到设备
                features = features.to(device)
                embeddings = embeddings.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(features, embeddings)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * features.size(0)  # 乘以batch_size以便后续平均

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        # 计算验证的平均损失和准确率
        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        try:
            val_roc_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_roc_auc = 0.0  # 如果只有一个类别，则ROC AUC无法计算

        print(f"\n验证损失: {val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        print(f"验证精确率: {val_precision:.4f}")
        print(f"验证召回率: {val_recall:.4f}")
        print(f"验证F1分数: {val_f1:.4f}")
        print(f"验证ROC AUC: {val_roc_auc:.4f}")

        # 学习率调度器步进
        scheduler.step(val_loss)

        # 早停机制
        if early_stopping.step(val_loss, model):
            print("触发早停机制！")
            break

    # 训练结束后加载最佳模型
    if early_stopping.best_model_wts is not None:
        model.load_state_dict(early_stopping.best_model_wts)
        final_model_path = os.path.join(model_save_dir, 'final_best_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"最佳模型已保存到 {final_model_path}")

if __name__ == '__main__':
    main()
