
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import laion_clap
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 自定义数据集类
class AudioClassificationDataset(Dataset):
    def __init__(self, root_dir, max_samples=None):
        """
        Args:
            root_dir (string): 数据根目录，包含多个子文件夹
            model: CLAP模型实例
            max_samples (int): 最大样本数（用于调试）
        """
        self.root_dir = root_dir
        self.audio_files = []
        self.labels = []
        
        # 扫描目录结构
        self._scan_directory()
        
        # 限制样本数量（用于调试）
        if max_samples:
            self.audio_files = self.audio_files[:max_samples]
            self.labels = self.labels[:max_samples]
        
        print(f"数据集加载完成，共 {len(self.audio_files)} 个样本")
    
    def _scan_directory(self):
        """扫描目录结构，收集音频文件和标签"""
        subdirs = [d for d in os.listdir(self.root_dir) 
                  if os.path.isdir(os.path.join(self.root_dir, d))]
        
        
        print(f"发现 {len(subdirs)} 个子文件夹")
        
        # 收集所有音频文件
        for subdir in subdirs:
            subdir_path = os.path.join(self.root_dir, subdir)       
            label_bin_path = os.path.join(self.root_dir, subdir, "label_binary.txt") 

            # 读抑郁症二分类label
            with open(label_bin_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()  # 读取第一行并去除空白字符
                label = int(first_line)  # 转换为整数

            # 支持常见的音频格式
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            for file in os.listdir(subdir_path):
                if any(file.lower().endswith(ext) for ext in audio_extensions) and (file.split('.')[0])[-3:] == "out":
                    audio_path = os.path.join(subdir_path, file)
                    self.audio_files.append(audio_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio = self.audio_files[idx]
        label = self.labels[idx]
        return audio, label