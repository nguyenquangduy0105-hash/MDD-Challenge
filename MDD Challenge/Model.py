#----------------Import Wav2vec2----------------
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyarrow as pa
from datasets import Dataset, Features, Value, Audio
import pandas as pd
import os
from torch import nn


class wav2vec2(nn.Module):
    def __init__(self, target_layers, device, model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.target_layers = target_layers
        self.device = device

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            all_layers = outputs.hidden_states
            selected_features = [all_layers[i] for i in self.target_layers]
            if len(selected_features) == 1:
                combined_features = selected_features[0]
            else:
                combined_features = torch.stack(selected_features, dim=0).mean(dim=0)

        return combined_features


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
# Check số lượng layer và số chiều ẩn (hidden size)
print(f"Số layer: {model.config.num_hidden_layers}")
print(f"Hidden size (Số đặc trưng): {model.config.hidden_size}")
print("Wav2Vec2" in model.__class__.__name__)

#----------------Acoustic  Feature Embedding----------------
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class AudioFeatureEmbedding(nn.Module):
    def __init__(self, device, drop_out=0.1):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=(1, 2)),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )

        self.bilstm1 = nn.LSTM(input_size=81, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True,
                               dropout=drop_out)
        self.rnn_norm1 = nn.LayerNorm(128)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True,
                               dropout=drop_out)
        self.rnn_norm2 = nn.LayerNorm(256)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = self.cnn_stack(x)

        B, C, Fe, T = x.shape
        x = x.view(B, C * Fe, T).contiguous()

        x = x.transpose(1, 2)
        x = x.contiguous()

        h_a, _ = self.bilstm1(x)
        h_a = self.rnn_norm1(h_a)

        h_a, _ = self.bilstm2(h_a)
        h_a = self.rnn_norm2(h_a)

        return h_a


class APLFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.win_length = 400
        self.hop_length = 160
        self.n_fft = 512

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=self.n_fft // 2 + 1
        )

    def forward(self, waveform):

        window = torch.hann_window(self.win_length, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True
        )
        spec = torch.abs(spec)

        energy = torch.sum(spec, dim=1)
        log_energy = torch.log(torch.clamp(energy, min=1e-5)).unsqueeze(1)

        mel_features = self.mel_scale(spec)
        log_mel = torch.log(torch.clamp(mel_features, min=1e-5))

        fbanks_81 = torch.cat([log_mel, log_energy], dim=1)

        fbanks_81 = fbanks_81.transpose(1, 2)

        return fbanks_81


class AudioEncoder(nn.Module):
    def __init__(self, device, n_mels=80):
        super().__init__()
        self.fbanks = APLFeatureExtractor(n_mels=n_mels)
        self.embedding_block = AudioFeatureEmbedding(device=device)

        self.device = device
        self.to(device)

    def forward(self, x, target_len):
        x = x.to(self.device)
        features = self.fbanks(x)
        output = self.embedding_block(features)

        output = output.transpose(1, 2)

        aligned_features = F.interpolate(
            output,
            size=target_len,
            mode='linear',
            align_corners=False
        )
        return aligned_features.transpose(1, 2)

#----------------Pitch Feature Embedding----------------
from torch import nn
import torch
import torchaudio
import librosa
import numpy
import torch.nn.functional as F


class PitchEncoder(nn.Module):
    def __init__(self, drop_out=0.1):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv1d(in_channels=250, out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 160),
            nn.Dropout(drop_out),

            nn.Conv1d(in_channels=160, out_channels=80, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 80),
            nn.Dropout(drop_out)
        )

        self.bilstm1 = nn.LSTM(input_size=80, hidden_size=64, bidirectional=True, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(128)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(256)

    def forward(self, au_input, target_len):
        if au_input.shape[2] != 250:
            print(f"CẢNH BÁO: Shape bị sai!, shape hiện tại : {au_input.shape}")
        x = au_input.transpose(1, 2)
        x = self.cnn_stack(x)

        x = x.transpose(1, 2)

        p, _ = self.bilstm1(x)
        p = self.norm1(p)
        p, _ = self.bilstm2(p)
        p = self.norm2(p)

        p = p.transpose(1, 2)
        aligned_features = F.interpolate(
            p,
            size=target_len,
            mode='linear',
            align_corners=False
        )
        aligned_features = aligned_features.transpose(1, 2)

        return aligned_features

#----------------Phonetic Feature and Liguistic Embedding----------------
from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding,self).__init__(vocab_size, d_model, padding_idx=1)


class PhoneticFeatureEmbedding(nn.Module):
    def __init__(self, drop_out=0.1):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )

        self.rnn_stack = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.rnn_norm = nn.LayerNorm(768)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn_stack(x) 
        x = x.squeeze(1)

        h_p, _ = self.rnn_stack(x)
        h_p = self.rnn_norm(h_p)

        return h_p


w2v2 = wav2vec2(target_layers=[9], device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


class PhoneEncoder(nn.Module):
    def __init__(self, device, model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        super().__init__()
        self.backbone = wav2vec2(target_layers=[9], device=device)
        self.embedding = PhoneticFeatureEmbedding()

    def forward(self, au_input):
        features = w2v2(au_input)
        if torch.isnan(features).any():
            print(features)
            raise ValueError("Dừng train do bị NaN", flush=True)
        h_p = self.embedding(features)

        return h_p


class CanonPhoEncode(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=1)

        self.bi_lstm = nn.LSTM(input_size=embed_dim,
                               hidden_size=hidden_dim,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, label_input):
        h_v, _ = self.bi_lstm(self.emb(label_input))
        h_k = h_v

        return h_k, h_v


class Decoder(nn.Module):
    def __init__(self, vocab_size, device, embed_dim=256, hidden_dim=512, drop_out=0.1):
        super().__init__()
        self.canon_phoneme_encoder = CanonPhoEncode(vocab_size, embed_dim, hidden_dim)
        self.au_enc = AudioEncoder(device=device)
        self.ph_enc = PhoneEncoder(device=device)
        self.p_enc = PitchEncoder()
        self.mha = nn.MultiheadAttention(embed_dim=1280, num_heads=4, device=device, batch_first=True)
        self.kv_proj = nn.Linear(1024, 1280)
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=2560, device=device),

            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(drop_out),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(drop_out),

            nn.Linear(512, vocab_size)
        )

    def forward(self, au_input, label_input, nccf):
        if nccf.device != au_input.device or nccf.dtype != au_input.dtype:
            nccf = nccf.to(device=au_input.device, dtype=au_input.dtype)

        h_p = self.ph_enc(au_input)
        p = self.p_enc(nccf, target_len=h_p.size(1))
        h_a = self.au_enc(au_input, target_len=h_p.size(1))
        h_cat = torch.cat((h_a, h_p, p), dim=-1)

        h_k, h_v = self.canon_phoneme_encoder(label_input)
        h_k = self.kv_proj(h_k)
        h_v = self.kv_proj(h_v)
        # Attention
        attn_output, _ = self.mha(query=h_cat, key=h_k, value=h_v.detach())
        attn_cat = torch.cat((attn_output, h_cat), dim=-1)

        attn_cat = self.classifier(attn_cat)
        if torch.isnan(h_cat).any():
            print(f"Audio is nan : {torch.isnan(h_a).any()}")
            print(f"Phoneme is nan : {torch.isnan(h_p).any()}")
        if torch.isnan(h_k).any():
            print(f"Canon key is nan: {torch.isnan(h_k).any()}")
        if torch.isnan(h_v).any():
            print(f"Canon value:{torch.isnan(h_v).any()}")
        return attn_cat