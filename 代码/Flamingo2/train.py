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
from dataloader import AudioClassificationDataset

import os
import yaml
import json
import argparse

import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from src.factory import create_model_and_transforms
from utils import Dict2Class, get_autocast, get_cast_dtype

autocast = get_autocast(
    "fp16" , cache_enabled = False
)

cast_dtype = get_cast_dtype("fp16")

config = yaml.load(open("/media/tongji/26_npj_audio/code/Flamingo2/audio-flamingo-audio_flamingo_2/inference_HF_pretrained/configs/inference.yaml"), Loader=yaml.FullLoader)

data_config = config['data_config']
model_config = config['model_config']
clap_config = config['clap_config']
args = Dict2Class(config['train_config'])

# 量化函数
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def get_num_windows(T, sr, clap_config):

    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):

    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    assert len(data.shape) == 1, data.shape
    return data

def load_audio(audio_path, clap_config):

    sr = 16000
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config) # hard code audio start to 0.0
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)
    
    return audio_clips, audio_embed_mask

# 简单的分类器
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# 训练函数
def train_model(flamingo_2_model, model, train_loader, val_loader, num_epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 记录训练过程
    train_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for audios, labels in train_pbar:
            labels =  labels.to(device)
            audio_clips, audio_embed_mask = load_audio(audios, clap_config)
            audio_clips = audio_clips.to(0, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = audio_embed_mask.to(0, dtype=cast_dtype, non_blocking=True)
            audio_x_out , _ = flamingo_2_model._encode_audio_x(audio_x=audio_clips, audio_x_mask=audio_embed_mask)
            audio_x_out = audio_x_out.mean(dim=1).squeeze(1)  
            audio_embed = audio_x_out.to(device)

            optimizer.zero_grad()
            outputs = model(audio_embed)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for audios, labels in val_pbar:
                labels =  labels.to(device)
                audio_clips, audio_embed_mask = load_audio(audios, clap_config)
                audio_clips = audio_clips.to(0, dtype=cast_dtype, non_blocking=True)
                audio_embed_mask = audio_embed_mask.to(0, dtype=cast_dtype, non_blocking=True)
                audio_x_out , _ = flamingo_2_model._encode_audio_x(audio_x=audio_clips, audio_x_mask=audio_embed_mask)
                audio_x_out = audio_x_out.mean(dim=1).squeeze(1)  
                audio_embed = audio_x_out.to(device)
                outputs = model(audio_embed)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证准确率: {val_acc:.2f}%, 验证F1: {val_f1:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_audio_classifier.pth')
            print(f'  ✓ 保存最佳模型，验证准确率: {val_acc:.2f}%')
        
        scheduler.step()
        print('-' * 60)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化flamingo_2模型


    flamingo_2_model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    device_id = 0
    flamingo_2_model = flamingo_2_model.to(device_id)
    flamingo_2_model.eval()

    # Load metadata
    with open("/media/tongji/26_npj_audio/model/Flamingo2/safe_ckpt/metadata.json", "r") as f:
        metadata = json.load(f)

    # Reconstruct the full state_dict
    state_dict = {}

    # Load each SafeTensors chunk
    for chunk_name in metadata:
        chunk_path = f"/media/tongji/26_npj_audio/model/Flamingo2/safe_ckpt/{chunk_name}.safetensors"
        chunk_tensors = load_file(chunk_path)

        # Merge tensors into state_dict
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = flamingo_2_model.load_state_dict(state_dict, False)
    
    # 初始化分类头
    print("正在初始化分类头...")

    model = AudioClassifier(input_dim=512, num_classes=2).to(device)

    # 数据路径 - 请根据您的实际路径修改
    data_root = "/media/tongji/26_npj_audio/data/EATD-Corpus/"  # 包含子文件夹的根目录
    
    # 创建数据集
    print("正在创建数据集...")
    train_dataset = AudioClassificationDataset(root_dir=data_root + 'train')
    val_dataset = AudioClassificationDataset(root_dir=data_root + 'validation')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    # 训练模型
    print("开始训练...")
    classifier = train_model(
        flamingo_2_model=flamingo_2_model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        lr=1e-3
    )
    
    print("训练完成！")

if __name__ == "__main__":
    main()