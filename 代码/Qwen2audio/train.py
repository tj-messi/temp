import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置可见的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from dataloader import AudioClassificationDataset

from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# 量化函数
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

# 简单的分类器
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_path": "/media/tongji/26_npj_audio/data/EATD-Corpus/train/2/negative_out.wav"},
    ]},
    {"role": "user", "content": "Determine whether the audio is from a depressed patient or a healthy individual."},
]

# 训练函数
def train_model(processor, Qwen2audio_model, model, train_loader, val_loader, num_epochs=50, lr=1e-3):
    # 定义设备
    qwen_device = torch.device('cuda:0')  # Qwen在GPU 0
    classifier_device = torch.device('cuda:1')  # 分类器在GPU 1
    
    print(f"Qwen2Audio 使用设备: {qwen_device}")
    print(f"分类器使用设备: {classifier_device}")
    
    # 移动模型到对应设备
    Qwen2audio_model = Qwen2audio_model.to(qwen_device)
    model = model.to(classifier_device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 记录训练过程
    train_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        Qwen2audio_model.eval()  # Qwen2Audio只用于特征提取，设为eval模式
        
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for audios, labels in train_pbar:
            labels = labels.to(classifier_device)

            # 提取音频特征 - 在GPU 0上处理
            with torch.no_grad():
                audio = []
                audio.append(librosa.load(
                    audios[0], 
                    sr=processor.feature_extractor.sampling_rate)[0]
                )
                inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True)
                inputs = inputs.to(qwen_device)  # 移动到Qwen的GPU
                
                input_features = inputs.input_features
                feature_attention_mask = inputs.feature_attention_mask

                # 获取音频特征
                audio_feat_lengths, audio_output_lengths = Qwen2audio_model.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                
                # 创建序列tensor
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=qwen_device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=Qwen2audio_model.audio_tower.conv1.weight.dtype, device=qwen_device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                # 前向传播获取音频特征
                audio_outputs = Qwen2audio_model.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = Qwen2audio_model.multi_modal_projector(selected_audio_feature)
                
                # 平均池化并移动到分类器的GPU
                audio_embed = audio_features.mean(dim=1).to(classifier_device)

            # 分类器训练 - 在GPU 1上
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
                labels = labels.to(classifier_device)
                
                # 提取验证集音频特征
                with torch.no_grad():
                    audio = []
                    audio.append(librosa.load(
                        audios[0], 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )
                    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True)
                    inputs = inputs.to(qwen_device)
                    
                    # 重复训练时的特征提取过程...
                    input_features = inputs.input_features
                    feature_attention_mask = inputs.feature_attention_mask
                    
                    audio_feat_lengths, audio_output_lengths = Qwen2audio_model.audio_tower._get_feat_extract_output_lengths(
                        feature_attention_mask.sum(-1)
                    )
                    batch_size, _, max_mel_seq_len = input_features.shape
                    max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                    
                    seq_range = (
                        torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=qwen_device)
                        .unsqueeze(0)
                        .expand(batch_size, max_seq_len)
                    )
                    lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                    padding_mask = seq_range >= lengths_expand

                    audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                        batch_size, 1, max_seq_len, max_seq_len
                    )
                    audio_attention_mask = audio_attention_mask_.to(
                        dtype=Qwen2audio_model.audio_tower.conv1.weight.dtype, device=qwen_device
                    )
                    audio_attention_mask[audio_attention_mask_] = float("-inf")

                    audio_outputs = Qwen2audio_model.audio_tower(input_features, attention_mask=audio_attention_mask)
                    selected_audio_feature = audio_outputs.last_hidden_state
                    audio_features = Qwen2audio_model.multi_modal_projector(selected_audio_feature)
                    audio_embed = audio_features.mean(dim=1).to(classifier_device)
                
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
    # 初始化模型
    print("正在加载Qwen2audio模型...")
    processor = AutoProcessor.from_pretrained("/media/tongji/26_npj_audio/model/Qwen2audio/qwen2_audio_instruct")
    Qwen2audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "/media/tongji/26_npj_audio/model/Qwen2audio/qwen2_audio_instruct", 
        device_map="cpu"  # 先加载到CPU，后面再手动分配
    )
    
    # 初始化分类器
    print("正在初始化分类头...")
    model = AudioClassifier(input_dim=4096, num_classes=2)
    
    # 数据路径
    data_root = "/media/tongji/26_npj_audio/data/EATD-Corpus/"
    
    # 创建数据集
    print("正在创建数据集...")
    train_dataset = AudioClassificationDataset(root_dir=data_root + 'train')
    val_dataset = AudioClassificationDataset(root_dir=data_root + 'validation')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    # 训练模型
    print("开始训练...")
    classifier = train_model(
        processor=processor,
        Qwen2audio_model=Qwen2audio_model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        lr=1e-3
    )
    
    print("训练完成！")

if __name__ == "__main__":
    main()