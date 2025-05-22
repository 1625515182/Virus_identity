import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import re

# 10. 自定义PyTorch数据集，整合频率矩阵、相对距离矩阵、DNABERT嵌入和注释标签
class KmerDatasetWithEmbeddingAndLabel(Dataset):
    def __init__(self, saved_data=None):
        """
        初始化数据集，加载已保存的数据
        """
        if saved_data is not None:
            self.feature_representations = saved_data['features']
            self.embeddings = saved_data['embeddings']
            self.sequence_ids = saved_data['sequence_ids']
            self.labels = saved_data['labels']
            self.label_mapping = saved_data['label_mapping']
            print("Loaded data from saved file.")
        else:
            raise ValueError("Saved data must be provided.")

    def filter_data(self):
        # 获取标签为2的索引
        valid_indices = [i for i, label in enumerate(self.labels) if label != 2]

        # 过滤数据，去除label为2的样本
        self.feature_representations = [self.feature_representations[i] for i in valid_indices]
        self.embeddings = [self.embeddings[i] for i in valid_indices]
        self.sequence_ids = [self.sequence_ids[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        print(f"Removed {len(self.labels) - len(valid_indices)} samples with label 2.")

    def __len__(self):
        return len(self.feature_representations)

    def __getitem__(self, idx):
        """
        返回特征矩阵、嵌入表示、标签和序列ID
        """
        feature = torch.tensor(self.feature_representations[idx], dtype=torch.float32)  # [64, 64, 2]
        embedding = self.embeddings[idx]  # [768]
        label = self.labels[idx]  # 整数标签
        seq_id = self.sequence_ids[idx]
        return feature, embedding, label, seq_id


class CustomDataLoader:
    def __init__(self, data_path, batch_size=50, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files = sorted(
            [f for f in os.listdir(data_path) if f.endswith('.pt')],
            key=lambda x: int(re.search(r'\d+', x).group())  # 按数字排序
        )
        self.num_batches = len(self.files) // self.batch_size  # 计算总批次
        self.current_batch = 0  # 当前批次索引

        if self.shuffle:
            np.random.shuffle(self.files)  # 如果需要，打乱文件顺序

    def __iter__(self):
        return self  # 返回自身，表示可迭代

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration  # 如果没有更多批次，停止迭代

        # 获取当前批次的文件
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_files = self.files[start_idx:end_idx]
        batch_data = []

        # 打印当前加载的批次文件信息
        print("                                                       ")
        print("-"*150)
        print(f"Loading batch {self.current_batch + 1} files: {batch_files}")

        for file in batch_files:
            file_path = os.path.join(self.data_path, file)
            data = torch.load(file_path)
            batch_data.append(data)

        # 合并一批的数据
        combined_data = {
            'features': np.concatenate([d['features'] for d in batch_data], axis=0),
            'embeddings': np.concatenate([d['embeddings'] for d in batch_data], axis=0),
            'sequence_ids': np.concatenate([d['sequence_ids'] for d in batch_data], axis=0),
            'labels': np.concatenate([d['labels'] for d in batch_data], axis=0),
            'label_mapping': batch_data[0]['label_mapping']  # 假设所有文件具有相同的标签映射
        }

        dataset = KmerDatasetWithEmbeddingAndLabel(saved_data=combined_data)
        dataset.filter_data()

        self.current_batch += 1  # 增加批次索引
        return dataset  # 返回当前批次的数据集

# CoordAtt模块
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = max(8, inp // reduction)  # 计算中间层的通道数

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 激活函数

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

# ScConv卷积模块
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num, "通道数必须大于等于分组数"
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))   # 偏置参数
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.group_num, -1)  # [N, G, C//G * H * W]
        mean = x.mean(dim=2, keepdim=True)  # [N, G, 1]
        std = x.std(dim=2, keepdim=True)    # [N, G, 1]
        x = (x - mean) / (std + self.eps)   # 标准化
        x = x.view(N, C, H, W)              # 恢复形状
        return x * self.weight + self.bias  # 应用权重和偏置

class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5, torch_gn: bool = False):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
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

class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super(CRU, self).__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
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
    def __init__(self, op_channel: int, group_num: int = 4, gate_treshold: float = 0.5, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super(ScConv, self).__init__()
        self.SRU = SRU(oup_channels=op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel=op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

# 11. 定义深度学习模型
class VirusClassifier(nn.Module):
    def __init__(self):
        super(VirusClassifier, self).__init__()

# 特征矩阵分支
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)  # 增加通道数
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)  # 降低Dropout率

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 增加全连接层的神经元数量
        self.dropout_fc1 = nn.Dropout(0.3)

        # 嵌入向量分支
        self.fc_embed1 = nn.Linear(768, 256)
        self.bn_embed1 = nn.BatchNorm1d(256)
        self.dropout_embed1 = nn.Dropout(0.2)
        self.fc_embed2 = nn.Linear(256, 128)
        self.bn_embed2 = nn.BatchNorm1d(128)
        self.dropout_embed2 = nn.Dropout(0.2)

        # 融合后全连接层
        self.fc_combined1 = nn.Linear(256 + 128, 128)  # 调整融合层的输入维度
        self.bn_combined1 = nn.BatchNorm1d(128)
        self.dropout_combined1 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 1)  # 二分类输出

    def forward(self, feature_matrix, embedding_vector):
        # 特征矩阵分支
        x = F.relu(self.bn1(self.conv1(feature_matrix)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)

        # 嵌入向量分支
        y = F.relu(self.bn_embed1(self.fc_embed1(embedding_vector)))
        y = self.dropout_embed1(y)
        y = F.relu(self.bn_embed2(self.fc_embed2(y)))
        y = self.dropout_embed2(y)

        # 融合
        combined = torch.cat((x, y), dim=1)
        combined = F.relu(self.bn_combined1(self.fc_combined1(combined)))
        combined = self.dropout_combined1(combined)
        # out = self.fc_out(combined)  # 输出logits
        out = torch.sigmoid(self.fc_out(combined))  # 输出概率
        return out

def main():
    data_path = r"/home/zhangjianpeng/Virus_identity/data/pytorch_dataset_100-400_val"  # 替换为数据文件夹路径
    batch_size = 1
    custom_loader = CustomDataLoader(data_path=data_path, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_classifier = VirusClassifier().to(device)
    # 加载保存的最佳模型
    model_classifier.load_state_dict(
        torch.load(r"/home/zhangjianpeng/Virus_identity/model/100-400.pth")
    )
    print("Loaded best model.")

    # 设置模型为评估模式
    model_classifier.eval()

    # 定义损失函数（虽然我们不需要它进行评估，但可以在需要时进行打印或调试）
    criterion = nn.BCELoss()  # 假设这是一个二分类任务

    # 初始化累积变量用于计算所有评估指标
    all_preds = []
    all_labels = []
    all_probs = []

    # 初始化累积变量用于计算平均准确率
    total_correct = 0
    total_samples = 0

    # 遍历测试集
    for batch_idx, dataset in enumerate(custom_loader):
        print(f"Processing batch {batch_idx + 1}/{custom_loader.num_batches}")
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 获取label_mapping
        label_mapping = dataset.label_mapping
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        # 存储当前批次的预测结果和标签
        batch_predictions = []
        batch_true_labels = []
        batch_seq_ids = []
        batch_prob_class0 = []  # 存储类别0的概率
        batch_prob_class1 = []  # 存储类别1的概率

        with torch.no_grad():
            for batch_features, batch_embeddings, batch_labels, batch_ids in val_loader:
                batch_features = batch_features.permute(0, 3, 1, 2).to(device).float()
                batch_embeddings = batch_embeddings.to(device).float()
                batch_labels = batch_labels.float().unsqueeze(1).to(device)

                # 推理
                outputs = model_classifier(batch_features, batch_embeddings)  # [batch_size, 1]
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item() * batch_features.size(0)

                # 获取预测概率
                prob_class1 = outputs.detach().cpu().numpy().flatten()  # 类别1的概率
                prob_class0 = 1 - prob_class1  # 类别0的概率
                batch_prob_class0.extend(prob_class0)
                batch_prob_class1.extend(prob_class1)

                # 获取预测标签
                preds = (prob_class1 > 0.5).astype(int)
                val_preds.extend(preds)
                val_labels.extend(batch_labels.detach().cpu().numpy().flatten())

                # 收集序列ID和预测结果
                batch_seq_ids.extend(batch_ids)
                batch_predictions.extend(preds)
                batch_true_labels.extend(batch_labels.detach().cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 更新累积变量
        correct = np.sum(np.array(val_preds) == np.array(val_labels))
        total_correct += correct
        total_samples += len(val_labels)

        # Accumulate all predictions and labels for overall metrics
        all_preds.extend(val_preds)
        all_labels.extend(val_labels)
        all_probs.extend(batch_prob_class1)

        # 输出当前批次的详细预测结果，包括两类概率
        print("\nCurrent Batch Inference Results:")
        print(f"{'Seq ID':<45} {'Prob_Class0':<15} {'Prob_Class1':<15} {'True Label':<25} {'Predicted Label':<25} {'Mapped Label'}")
        print("="*150)

        for seq_id, prob0, prob1, true_label, pred_label in zip(batch_seq_ids, batch_prob_class0, batch_prob_class1, batch_true_labels, batch_predictions):
            # 使用反转后的映射来获取类别名称
            mapped_label = reverse_label_mapping.get(int(pred_label), 'Unknown')
            print(f"{seq_id:<45} {prob0:<15.4f} {prob1:<15.4f} {int(true_label):<25} {pred_label:<25} {mapped_label}")

        print("\n" + "="*150 + "\n")  # 分隔不同批次的输出

    # 计算所有批次的整体评估指标
    if total_samples > 0:
        # Convert lists to numpy arrays for metric calculations
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds_np).ravel()

        # 计算敏感性 (Sensitivity, Recall for positive class)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 计算特异性 (Specificity, Recall for negative class)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 计算F1-score
        f1 = f1_score(all_labels_np, all_preds_np)

        # 计算AUC (ROC Curve Area)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc = float('nan')  # 如果只有一个类存在，AUC无法计算

        # 计算总体准确率
        accuracy = accuracy_score(all_labels_np, all_preds_np)

        print("Overall Evaluation Metrics:")
        print(f"Sensitivity (Sn): {sensitivity:.4f}")
        print(f"Specificity (Sp): {specificity:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC (ROC Curve Area): {auc:.4f}")
        print(f"Accuracy (ACC): {accuracy:.4f}")
    else:
        print("No samples were processed.")

    print("Inference completed.")

if __name__ == "__main__":
    main()
