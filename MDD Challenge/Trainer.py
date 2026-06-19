import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim import Adam
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import *

import gc

gc.collect()
torch.cuda.empty_cache()


pkl_path = '/kaggle/working/Encoded.pkl'
with open(pkl_path, 'rb') as f:
    samples = pickle.load(f)

vocab_path = 'vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
P2idx = vocab['p2idx']
idx2P = vocab['idx2p']


class L2ArcticDataset(Dataset):
    def __init__(self, data_list, sampler_rate=16000):
        self.data_list = data_list
        self.sampler_rate = sampler_rate
        self.dataset_root = "/kaggle/input/datasets/mapotofu41/en-mdd/EN_MDD/WAV"

        from transformers import Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        raw_path = sample['path']
        if not raw_path.endswith('.wav'):
            raw_path += '.wav'

        path = os.path.join(self.dataset_root, raw_path) if not raw_path.startswith('/kaggle/input') else raw_path

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sampler_rate:
            waveform = T.Resample(sr, self.sampler_rate)(waveform)

        waveform_np = waveform.squeeze().numpy()

        inputs = self.processor(
            waveform_np,
            sampling_rate=self.sampler_rate,
            return_tensors="pt"
        )

        input_values = inputs.input_values.squeeze(0)

        if torch.isnan(waveform).any():
            print("Waveform bị NAN ngay từ DataLoader!")
            waveform = torch.nan_to_num(waveform)
        return (
            input_values,
            torch.LongTensor(sample['transcript']),
            torch.LongTensor(sample['canonical'])
        )


def get_nccf(au_input, hop_length=160, win_length=400):
    if torch.is_tensor(au_input):
        au_input = au_input.detach().cpu().numpy()

    au_input = np.ascontiguousarray(au_input)

    frames = librosa.util.frame(au_input, hop_length=hop_length, frame_length=win_length)
    nccf_frames = []

    for i in range(frames.shape[1]):
        frame = frames[:, i]

        corr = librosa.autocorrelate(frame)

        energy = corr[0] if corr[0] > 1e-8 else 1.0
        normalized_corr = corr / energy

        nccf_frames.append(normalized_corr[:250])
    nccf_matrix = np.array(nccf_frames)

    nccf_tensor = torch.tensor(nccf_matrix, dtype=torch.float32)

    return nccf_tensor


pad_idx = P2idx['<pad>']


def collate_fn(batch):
    waveform = [item[0] for item in batch]
    transcript = [item[1] for item in batch]
    canonical = [item[2] for item in batch]
    nccf = []
    for wf in waveform:
        nccf.append(get_nccf(wf))

    nccf_padded = pad_sequence(nccf, batch_first=True, padding_value=0.0)
    waveform_padded = pad_sequence(waveform, batch_first=True, padding_value=0.0)
    trans_padded = pad_sequence(transcript, batch_first=True, padding_value=pad_idx)
    canon_padded = pad_sequence(canonical, batch_first=True, padding_value=pad_idx)

    input_lengths = torch.LongTensor([s.size(0) for s in waveform])
    label_lengths = torch.LongTensor([l.size(0) for l in transcript])

    return waveform_padded, trans_padded, canon_padded, input_lengths, label_lengths, nccf_padded


train_dataset = L2ArcticDataset(data_list=samples['train'])
dev_dataset = L2ArcticDataset(data_list=samples['dev'])

train_loader = DataLoader(train_dataset,
                          batch_size=12,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=4)
dev_loader = DataLoader(dev_dataset,
                        batch_size=12,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4)


def weight_init(m):
    if "wav2vec2" in m.__class__.__name__.lower() or "encoder" in m.__class__.__name__.lower():
        return
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Decoder(vocab_size=len(P2idx), device=device)
model.to(device)

model.mha.apply(weight_init)
model.classifier.apply(weight_init)
model.canon_phoneme_encoder.apply(weight_init)
model.ph_enc.embedding.apply(weight_init)
model.au_enc.apply(weight_init)

criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)

total_epochs = 20
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)


# ----------------------------Mixed Precision----------------------------------
scaler = torch.amp.GradScaler("cuda")


def train(model, iterator, optimizer, criterion, device, clip=1.0):
    step = 0
    model.train()
    train_loss = 0
    for batch_idx, (waveform, transcript, canon, input_lengths, label_lengths, nccf) in enumerate(iterator):
        waveform, transcript, canon, nccf = waveform.to(device), transcript.to(device), canon.to(device), nccf.to(
            device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            input_lengths = input_lengths.to(device)
            logits = model(waveform, canon, nccf)

            current_batch_size = logits.size(0)
            actual_time_steps = logits.size(1)

            logits = logits.transpose(0, 1)

            log_probs = F.log_softmax(logits, dim=2)

            actual_time_steps = logits.size(0)

            scaled_input_lengths = torch.clamp(input_lengths // 320, max=actual_time_steps).to(device)
            label_lengths = label_lengths.to(device)

            loss = criterion(log_probs, transcript, scaled_input_lengths, label_lengths)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        step += 1
        if step % 20 == 0: print(f"step: {step}")

    return train_loss / len(iterator)


def to_scalar(obj):
    if isinstance(obj, (list, tuple, torch.Tensor)):
        return to_scalar(obj[0])
    return int(obj)


def visualize_random_samples(model, dataset, device, num_samples=5, idx2p=None):
    model.eval()
    CTC_BLANK_ID = 0
    clean_target = []
    with torch.no_grad():
        for _ in range(num_samples):
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]

            batch = collate_fn([sample])
            waveform, transcript, canonical, input_lengths, label_lengths, nccf = batch
            waveform = waveform.to(device)
            canonical = canonical.to(device)
            transcript = transcript.to(device)
            nccf = nccf.to(device, dtype=waveform.dtype)

            logits = model(waveform, canonical, nccf)

            pred_phonemes_raw, pred_ID = ctc_greedy_decode(logits[0], idx2p, blank_token=CTC_BLANK_ID)

            clean_pred = [p for p in pred_phonemes_raw]

            trans = transcript[0]
            canon = canonical[0]

            print(f"Sample index: {idx}")
            print(f"Canon : {canon}")
            print(f"Transcript : {trans}")
            print(f"Output : {torch.tensor(pred_ID)}")
            print()


def validation(model, iterator, device, num_samples=3, idx2p=idx2P):
    model.eval()
    val_loss = 0
    samples_logged = 0
    with torch.no_grad():
        for batch_idx, (waveform, transcript, canon, input_lengths, label_lengths, nccf) in enumerate(iterator):
            waveform, transcript, canon, nccf = waveform.to(device), transcript.to(device), canon.to(device), nccf.to(
                device)

            input_lengths = input_lengths.to(device)
            logits = model(waveform, canon, nccf)

            current_batch_size = logits.size(0)
            actual_time_steps = logits.size(1)

            ctc_logits = logits.transpose(0, 1)
            ctc_log_probs = F.log_softmax(ctc_logits, dim=2)

            scaled_input_lengths = torch.clamp(input_lengths // 320, max=actual_time_steps).to(device)
            label_lengths = label_lengths.to(device)

            loss = criterion(ctc_log_probs, transcript, scaled_input_lengths, label_lengths)
            val_loss += loss.item()

    return val_loss / len(iterator)


def ctc_greedy_decode(probs, idx2p, blank_token=0):
    best_path = torch.argmax(probs, dim=-1).cpu().numpy()
    print(f"best_path: {best_path}")

    collapsed_path = []
    last_token = None
    for token in best_path:
        if token != last_token:
            collapsed_path.append(token)
            last_token = token

    phonemes = [idx2p[token] for token in collapsed_path if token != blank_token]

    return phonemes, collapsed_path


best_loss = float('inf')
patience = 5
counter = 0
train_loss = 0
test_loss = 0

for epoch in range(100):
    train_loss = train(model, train_loader, optimizer, criterion, device=device)
    val_loss = validation(model, dev_loader, device=device)

    visualize_random_samples(model, dev_dataset, device=device, idx2p=idx2P)

    scheduler.step()

    print(val_loss)
    if best_loss is None:
        best_loss = float('inf')

    if val_loss is not None and val_loss < best_loss:
        best_loss = val_loss
        upload_best_model(model=model, epoch=epoch, val_loss=val_loss, username='nguyenquangduy15',
                          dataset_name="papl-train-model")
        print(f"Đã lưu model tốt nhất tại epoch với loss: {best_loss:.4f}")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early Stopping triggered")
        break

    gc.collect()
    torch.cuda.empty_cache()