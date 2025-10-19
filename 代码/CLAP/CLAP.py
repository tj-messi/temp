import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import librosa
import torch
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt("/media/tongji/26_npj_audio/model/CLAP/630k-audioset-best.pt") # download the default pretrained checkpoint.

# Directly get audio embeddings from audio files
audio_file = [
    '/media/tongji/26_npj_audio/data/EATD-Corpus/train/1/negative_out.wav',
    '/media/tongji/26_npj_audio/data/EATD-Corpus/train/1/neutral_out.wav'
]
audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
print(audio_embed[:,-20:])
print(audio_embed.shape)