import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioEncoder
import torch

processor = AutoProcessor.from_pretrained("/media/tongji/26_npj_audio/model/Qwen2audio/qwen2_audio_instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("/media/tongji/26_npj_audio/model/Qwen2audio/qwen2_audio_instruct", device_map="auto")
# Encoder = Qwen2AudioEncoder.from_pretrained("/media/tongji/26_npj_audio/model/Qwen2audio/qwen2audio-encoder", device_map="cpu")

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_path": "/media/tongji/26_npj_audio/data/EATD-Corpus/train/2/negative_out.wav"},
    ]},
    {"role": "user", "content": "Guess the gender"},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(librosa.load(
                    ele['audio_path'], 
                    sr=processor.feature_extractor.sampling_rate)[0]
                )

inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
inputs = inputs.to('cuda')

# generate_ids = model.generate(**inputs, max_length=256)
# generate_ids = generate_ids[:, inputs.input_ids.size(1):]

# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(response)

input_features = inputs.input_features
feature_attention_mask = inputs.feature_attention_mask

input_embedding = model.get_input_embeddings()(inputs.input_ids)
print(input_embedding.shape)

audio_feat_lengths, audio_output_lengths = model.audio_tower._get_feat_extract_output_lengths(
    feature_attention_mask.sum(-1)
)
batch_size, _, max_mel_seq_len = input_features.shape
max_seq_len = (max_mel_seq_len - 2) // 2 + 1
# Create a sequence tensor of shape (batch_size, max_seq_len)
seq_range = (
    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
    .unsqueeze(0)
    .expand(batch_size, max_seq_len)
)
lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
# Create mask
padding_mask = seq_range >= lengths_expand

audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
    batch_size, 1, max_seq_len, max_seq_len
)
audio_attention_mask = audio_attention_mask_.to(
    dtype=model.audio_tower.conv1.weight.dtype, device=model.audio_tower.conv1.weight.device
)
audio_attention_mask[audio_attention_mask_] = float("-inf")

audio_outputs = model.audio_tower(input_features, attention_mask=audio_attention_mask)
selected_audio_feature = audio_outputs.last_hidden_state
audio_features = model.multi_modal_projector(selected_audio_feature)

print(audio_features.shape)