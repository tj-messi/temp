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

# 训练函数
def train_model(CLAP_model, model, train_loader, val_loader, num_epochs=50, lr=1e-3):
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
            audio_embed = CLAP_model.get_audio_embedding_from_filelist(x = audios, use_tensor=False)
            audio_embed = torch.from_numpy(audio_embed).to(device)

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
                audio_embed = CLAP_model.get_audio_embedding_from_filelist(x = audios, use_tensor=False)
                audio_embed = torch.from_numpy(audio_embed).to(device)
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

    # 初始化CLAP模型
    print("正在加载CLAP模型...")
    CLAP_model = laion_clap.CLAP_Module(enable_fusion=False)
    CLAP_model.load_ckpt("/media/tongji/26_npj_audio/model/CLAP/630k-audioset-best.pt")
    CLAP_model.eval()
    CLAP_model.to(device)
    
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
        CLAP_model=CLAP_model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        lr=1e-3
    )
    
    print("训练完成！")

if __name__ == "__main__":
    main()