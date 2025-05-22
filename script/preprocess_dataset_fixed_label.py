import numpy as np
from itertools import product
from collections import defaultdict
import bisect
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertConfig
import glob
import re
import multiprocessing
from torch.cuda.amp import autocast

# 1. 定义不确定碱基字典
dictuncertain = {
    'N': 'G',
    'X': 'A',
    'H': 'T',
    'M': 'C',
    'K': 'G',
    'D': 'A',
    'R': 'G',
    'Y': 'T',
    'S': 'C',
    'W': 'A',
    'B': 'C',
    'V': 'G',
}


# 2. 生成所有可能的三核苷酸组合及其索引映射
def generate_kmer_mapping():
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_list = [''.join(p) for p in product(nucleotides, repeat=3)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(kmer_list)}
    index_to_kmer = {i: kmer for i, kmer in enumerate(kmer_list)}
    return kmer_list, kmer_to_index, index_to_kmer


# 3. 替换不确定碱基
def replace_uncertain_bases(dna_sequence, dictuncertain):
    replaced_sequence = []
    for base in dna_sequence.upper():
        if base in dictuncertain:
            replaced_sequence.append(dictuncertain[base])
        else:
            replaced_sequence.append(base)
    return ''.join(replaced_sequence)


# 4. 解析DNA序列为3-mers列表并记录位置
def parse_kmers(dna_sequence, kmer_to_index):
    S = []
    positions = defaultdict(list)  # 记录每个kmer的出现位置
    for i in range(len(dna_sequence) - 2):
        kmer = dna_sequence[i:i + 3]
        if kmer in kmer_to_index:
            S.append(kmer)
            positions[kmer_to_index[kmer]].append(i)
        else:
            S.append(None)  # 无效的kmer（理论上应不存在）
    return S, positions


# 5. 计算频率矩阵
def compute_frequency_matrix(S, kmer_to_index):
    frequency_matrix = np.zeros((64, 64), dtype=int)
    for i in range(len(S) - 1):
        kmer1 = S[i]
        kmer2 = S[i + 1]
        if kmer1 and kmer2:
            idx1 = kmer_to_index[kmer1]
            idx2 = kmer_to_index[kmer2]
            frequency_matrix[idx1][idx2] += 1
    return frequency_matrix


# 6. 计算相对距离矩阵（使用最近邻距离的平均值）
def compute_distance_matrix(S, kmer_to_index, positions):
    # 初始化距离矩阵为-1，表示未共现
    distance_matrix = np.full((64, 64), -1.0)
    for i in range(64):
        for j in range(64):
            if not positions[i] or not positions[j]:
                continue  # 保持为-1
            pos_i = positions[i]
            pos_j = positions[j]
            distances = []
            for pos in pos_i:
                idx = bisect.bisect_left(pos_j, pos)
                candidates = []
                if idx < len(pos_j):
                    candidates.append(abs(pos_j[idx] - pos))
                if idx > 0:
                    candidates.append(abs(pos - pos_j[idx - 1]))
                if candidates:
                    distances.append(min(candidates))
            if distances:
                distance_matrix[i][j] = sum(distances) / len(distances)
    return distance_matrix

def compute_weighted_distance_matrix(kmer_to_index, positions, frequency_matrix):
    num_kmers = len(kmer_to_index)
    distance_matrix_weighted = np.zeros((num_kmers, num_kmers), dtype=float)

    for i in range(num_kmers):
        for j in range(num_kmers):
            if not positions[i] or not positions[j]:
                continue
            pos_i = positions[i]
            pos_j = positions[j]
            distances = []
            for pos in pos_i:
                idx = bisect.bisect_left(pos_j, pos)
                candidates = []
                if idx < len(pos_j):
                    candidates.append(abs(pos_j[idx] - pos))
                if idx > 0:
                    candidates.append(abs(pos - pos_j[idx - 1]))
                if candidates:
                    distances.append(min(candidates))
            if distances:
                avg_distance = np.mean(distances)
                distance_matrix_weighted[i][j] = avg_distance * frequency_matrix[i][j]

    return distance_matrix_weighted
# 7. 创建双通道特征表示
def create_feature_representation(frequency_matrix, distance_matrix, sequence_length):
    # 处理距离矩阵中的-1值，可以选择替换为序列长度
    distance_matrix_clean = np.copy(distance_matrix)
    distance_matrix_clean[distance_matrix_clean == -1] = sequence_length

    # 正则化距离矩阵，例如将其缩放到0-1之间
    max_distance = np.max(distance_matrix_clean)
    if max_distance > 0:
        distance_matrix_normalized = distance_matrix_clean / max_distance
    else:
        distance_matrix_normalized = distance_matrix_clean

    # 正则化频率矩阵
    max_frequency = np.max(frequency_matrix)
    if max_frequency > 0:
        frequency_matrix_normalized = frequency_matrix / max_frequency
    else:
        frequency_matrix_normalized = frequency_matrix

    # 堆叠成双通道
    feature_representation = np.stack((frequency_matrix_normalized, distance_matrix_normalized), axis=-1)
    return feature_representation


# 8. 加载DNABERT模型和分词器
def load_dnabert_model(model_path):
    """
    加载DNABERT模型和分词器
    """
    config = BertConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # 设置分词器的多线程
    num_threads = multiprocessing.cpu_count()
    tokenizer.truncate_sequences = True
    tokenizer.padding_side = "right"
    tokenizer.num_threads = num_threads
    tokenizer.max_length = 600  # 设置一个固定的最大长度

    return model, tokenizer


# 9. 获取序列嵌入（优化版，添加每批输出和具体嵌入打印）
def get_sequence_embeddings(sequences, sequence_ids, model, tokenizer, device='cpu', batch_size=32):
    """
    生成序列嵌入，支持批处理和混合精度
    Args:
        sequences (List[str]): DNA序列列表
        sequence_ids (List[str]): 序列对应的ID列表
        model: 预训练的DNABERT模型
        tokenizer: 对应的分词器
        device (str): 'cpu' 或 'cuda'
        batch_size (int): 每批处理的序列数
    Returns:
        torch.Tensor: 嵌入向量
    """
    model.to(device)
    model.eval()
    embeddings = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_sequence_ids = sequence_ids[i:i + batch_size]
            inputs = tokenizer(
                batch_sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=tokenizer.max_length,
            ).to(device)

            if device.type == 'cuda':
                with autocast():
                    outputs = model(**inputs)
                    batch_embeddings = outputs[1]  # 修改这里，从元组中获取第一个元素
                    # batch_embeddings = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
            else:
                outputs = model(**inputs)
                batch_embeddings = outputs[1]  # 修改这里，从元组中获取第一个元素
                # batch_embeddings = torch.mean(hidden_states, dim=1)

            batch_embeddings = batch_embeddings.cpu()
            embeddings.append(batch_embeddings)

            # 输出每个序列的嵌入向量
            # for seq_id, embedding in zip(batch_sequence_ids, batch_embeddings):
            #     print(f"序列ID: {seq_id}")
                # print(f"嵌入向量: {embedding.numpy()}\n")

            # 输出每批处理的信息
            current_batch = (i // batch_size) + 1
            print(f"已处理批次 {current_batch} / {num_batches}\n")

    return torch.cat(embeddings, dim=0)


# 10. 自定义PyTorch数据集
class KmerDatasetWithEmbeddingAndLabel(Dataset):
    def __init__(self, fasta_dir, global_label, dictuncertain, kmer_to_index, index_to_kmer, model, tokenizer, save_dir,
                 device='cpu'):
        """
        global_label: 全局标签字符串，应用于所有FASTA文件中的序列
        """
        self.device = device
        self.save_dir = save_dir
        self.kmer_to_index = kmer_to_index
        self.index_to_kmer = index_to_kmer
        self.model = model
        self.tokenizer = tokenizer
        self.global_label = global_label

        # 创建标签映射（字符串标签到整数索引）
        self.label_mapping = {'Eukaryotes Virus': 0, 'Prokaryotes Virus': 1, 'Unknown': 2}
        if self.global_label not in self.label_mapping:
            raise ValueError(f"指定的全局标签'{self.global_label}'不在预定义的标签映射中: {self.label_mapping}")
        print(f"标签映射: {self.label_mapping}")

        self.model.eval()
        self.model.to(self.device)

        # 遍历FASTA文件目录下的所有FASTA文件，并按文件名排序
        fasta_files = sorted(
            glob.glob(os.path.join(fasta_dir, "*.fasta")) +
            glob.glob(os.path.join(fasta_dir, "*.fa")) +
            glob.glob(os.path.join(fasta_dir, "*.fna")),
            key=self.extract_sequence_number  # 按照序号排序
        )

        # 处理每个FASTA文件
        for idx, fasta_file in enumerate(fasta_files, start=1):
            print(f"正在处理FASTA文件: {fasta_file}")

            sequences = []
            seq_ids = []
            labels = []
            feature_representations = []

            for record in SeqIO.parse(fasta_file, "fasta"):
                try:
                    sequence = str(record.seq)
                    seq_id_full = record.id  # e.g., "LH00391:53:22G23FLT3:4:1101:39677:1112"

                    # 替换不确定碱基
                    sequence_replaced = replace_uncertain_bases(sequence, dictuncertain)

                    # 解析序列
                    S, positions = parse_kmers(sequence_replaced, kmer_to_index)

                    # 计算频率矩阵
                    frequency_matrix = compute_frequency_matrix(S, kmer_to_index)

                    # 计算相对距离矩阵
                    # distance_matrix = compute_distance_matrix(S, kmer_to_index, positions)
                    weighted_distance_matrix = compute_weighted_distance_matrix(kmer_to_index, positions,frequency_matrix)

                    # 创建双通道特征表示
                    feature_representation = create_feature_representation(frequency_matrix, weighted_distance_matrix,
                                                                           len(sequence_replaced))

                    sequences.append(sequence_replaced)
                    seq_ids.append(seq_id_full)
                    labels.append(self.label_mapping[self.global_label])
                    feature_representations.append(feature_representation)

                except Exception as e:
                    print(f"处理序列 '{record.id}' 时出错: {e}")
                    continue

            # 生成嵌入
            if sequences:
                print("生成DNABERT嵌入...\n")
                embeddings = get_sequence_embeddings(
                    sequences,
                    seq_ids,
                    self.model,
                    self.tokenizer,
                    device=self.device,
                    batch_size=100
                )

                # # 保存每个FASTA文件的数据为临时文件
                # temp_data = {
                #     'features': feature_representations,
                #     'embeddings': embeddings,
                #     'sequence_ids': seq_ids,
                #     'labels': labels,
                #     'label_mapping': self.label_mapping
                # }
                # temp_file = os.path.join(
                #     self.save_dir,
                #     os.path.basename(fasta_file).replace('.fasta', '.pt').replace('.fa', '.pt').replace('.fna', '.pt')
                # )
                # torch.save(temp_data, temp_file)
                # print(f"已保存临时数据到 {temp_file}\n")
                        # 使用自定义名称和序号生成文件名
                custom_name = f"Prokaryotes Virus{idx}"  # 加入从1开始的序号

                # 保存文件
                temp_file = os.path.join(
                    self.save_dir,
                    custom_name + '.pt'  # 使用自定义名称和.pt后缀
                )

                torch.save({
                    'features': feature_representations,
                    'embeddings': embeddings,
                    'sequence_ids': seq_ids,
                    'labels': labels,
                    'label_mapping': self.label_mapping
                }, temp_file)
                print(f"已保存临时数据到 {temp_file}\n")

    def extract_sequence_number(self, filepath):
        """提取文件名中任意字母后面的数字部分用于排序"""
        basename = os.path.basename(filepath)  # 提取文件名
        matches = re.findall(r'[A-Za-z]+(\d+)', basename)  # 查找所有匹配
        if matches:
            # 选择最后一个匹配的数字
            return int(matches[-1])
        else:
            print(f"警告: 文件名 '{basename}' 不包含字母后面的数字，将被排序为最后")
            return float('inf')  # 将不匹配的文件排到最后

    def __len__(self):
        # 根据实际需求实现，例如统计所有序列的数量
        pass

    def __getitem__(self, idx):
        # 根据实际需求实现，例如返回特定索引的数据
        pass


# 主函数
def main():
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 保存处理好的对应的pytorch数据集格式的目录
    save_dir = r"/home/zhangjianpeng/Virus_identity/data/pig/pytorch"
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir)
    # fasta文件目录
    fasta_dir = r"/home/zhangjianpeng/Virus_identity/data/pig/prokar"
    # 手动指定全局标签
    # 可选标签: 'Eukaryotes Virus', 'Prokaryotes Virus', 'Unknown'
    global_label = 'Prokaryotes Virus'  # 在此处更改标签


    # 生成k-mer映射
    kmer_list, kmer_to_index, index_to_kmer = generate_kmer_mapping()

    # 加载DNABERT模型和分词器
    model_path = r"/home/zhangjianpeng/Virus_identity/My_DNABERT/trained_model"
    model, tokenizer = load_dnabert_model(model_path)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 创建数据集
    try:
        dataset = KmerDatasetWithEmbeddingAndLabel(
            fasta_dir=fasta_dir,
            global_label=global_label,
            dictuncertain=dictuncertain,
            kmer_to_index=kmer_to_index,
            index_to_kmer=index_to_kmer,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            device=device  # 确保设备传递正确
        )
    except ValueError as ve:
        print(f"标签错误: {ve}")


if __name__ == "__main__":
    main()
