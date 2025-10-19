## 代码架构

code
  CLAP
  Flamingo2
  Qwen2audio 
data
  EATD-Corpus
    train
        1
            label_binary.txt
            negative_out.wav
            neutral_out.wav
            positive_out.wav
        2
        3
    validation
model
  CLAP
  Flamingo2
  Qwen2audio 


## 数据准备

把EATD-Corpus的源数据下载到data/目录下

格式和代码架构展示的一样

## 预训练模型准备

### CLAP

从官方链接下载：630k-audioset-fusion-best.pt

### Qwen2audio

Huggingface上下载：qwen2_audio_instruct

### Flamingo2

从魔搭社区下载：nvidia-Flamingo2-audio

## 训练

### CLAP

python code/CLAP/train.py

### Qwen2audio

python code/Qwen2audio/train.py

### Flamingo2

python code/Flamingo2/train.py

## 实验结果

| Model     | Result (ACC) |
| CLAP      |    86.6      |
| Qwen2     |    87.6      |
| Flamingo2 |    85.4      |