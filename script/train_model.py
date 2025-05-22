import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import re


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
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        """
        :param patience: 如果验证准确度在连续的patience个epoch中没有提升，则提前停止训练。
        :param delta: 如果验证准确度的变化小于delta，则不认为是改善。
        :param path: 保存最佳模型的路径。
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_acc = None
        self.best_model_wts = None
        self.path = path

    def step(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_model_wts = model.state_dict()
        elif val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.best_model_wts = model.state_dict()
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("触发早停机制！")
            torch.save(self.best_model_wts, self.path)  # 保存最好的模型
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


# 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    data_path = r"/home/zhangjianpeng/Virus_identity/data/pytorch_dataset_1200-1800"

    # 获取所有数据文件的列表
    all_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith('.pt')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )

    # 打乱文件顺序
    random.shuffle(all_files)

    # 划分为训练集和验证集文件
    train_size = int(0.8 * len(all_files))
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    print(f"总文件数: {len(all_files)}")
    print(f"训练文件数: {len(train_files)}")
    print(f"验证文件数: {len(val_files)}")

    # 创建训练和验证的 CustomDataLoader
    train_loader = CustomDataLoader(data_path, files=train_files, batch_size=20, shuffle=True)
    val_loader = CustomDataLoader(data_path, files=val_files, batch_size=20, shuffle=False)

    # 定义模型保存目录
    model_save_dir = r"/home/zhangjianpeng/Virus_identity/model"
    os.makedirs(model_save_dir, exist_ok=True)  # 如果目录不存在，则创建

    # 初始化模型、优化器、损失函数、学习率调度器、早停机制
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    model = VirusClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 设置最佳模型保存路径
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')
    early_stopping = EarlyStopping(patience=5, delta=0.001, path=best_model_path)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        model.train()
        epoch_running_loss = 0.0
        epoch_total = 0

        # 训练循环
        for batch_idx, dataset in enumerate(train_loader):
            print(f"\n处理训练批次 {batch_idx + 1}/{len(train_loader)}")
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # 初始化外部批次的损失和正确预测数
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0

            for features, embeddings, labels, seq_ids in data_loader:
                features = features.to(device)
                embeddings = embeddings.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(features, embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 累加损失
                batch_loss += loss.item() * features.size(0)  # 乘以batch_size以便后续平均

                # 计算预测结果
                preds = torch.sigmoid(outputs) >= 0.5
                batch_correct += (preds.float() == labels).sum().item()
                batch_total += labels.size(0)

            # 计算该外部批次的平均损失和准确率
            avg_batch_loss = batch_loss / batch_total
            batch_accuracy = batch_correct / batch_total

            print(
                f"  批次 {batch_idx + 1}/{len(train_loader)} - 平均损失: {avg_batch_loss:.4f} - 准确率: {batch_accuracy:.4f}")

            # 累加到epoch的总损失和总样本数
            epoch_running_loss += batch_loss
            epoch_total += batch_total

        # 计算并打印本epoch的平均训练损失
        epoch_loss = epoch_running_loss / epoch_total
        print(f"\n本epoch训练平均损失: {epoch_loss:.4f}")

        # 验证循环
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for batch_idx, dataset in enumerate(val_loader):
                print(f"\n处理验证批次 {batch_idx + 1}/{len(val_loader)}")
                data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

                for features, embeddings, labels, seq_ids in data_loader:
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
        if early_stopping.step(val_accuracy, model):
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
