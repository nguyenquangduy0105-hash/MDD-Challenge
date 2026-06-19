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

import gc

gc.collect()
torch.cuda.empty_cache()

print(f"📍 Script đang chạy tại: {os.getcwd()}")


pkl_path = '/kaggle/working/Encoded.pkl'
with open(pkl_path, 'rb') as f:
    samples = pickle.load(f)

# Load vocab
vocab_path = 'vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
P2idx = vocab['p2idx']
idx2P = vocab['idx2p']


class L2ArcticDataset(Dataset):
    def __init__(self, data_list, sampler_rate=16000):
        self.data_list = data_list
        self.sampler_rate = sampler_rate
        self.dataset_root = "/kaggle/input/datasets/nguyenquangduy15/challenge-test/MDD-Challenge-2025-public-test (1)/MDD-Challenge-2025-public-test"

        from transformers import Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        raw_path = sample['path']
        if not raw_path.endswith('.wav'):
            raw_path += '.wav'

        path = os.path.join(self.dataset_root, raw_path)

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
            torch.LongTensor(sample['canonical']),
            torch.LongTensor(sample['transcript'])
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



test_dataset = L2ArcticDataset(data_list=samples['test'])

test_loader = DataLoader(test_dataset,
                         batch_size=16,
                         shuffle=False,
                         collate_fn=collate_fn,
                         num_workers=4,
                         pin_memory=True,
                         persistent_workers=True,
                         prefetch_factor=4)


def get_clean_sequence(tensor_ids, blank_token=0):

    ids = tensor_ids.tolist() if hasattr(tensor_ids, 'tolist') else list(tensor_ids)

    collapsed = []
    if len(ids) > 0:
        collapsed.append(ids[0])
        for i in range(1, len(ids)):
            if ids[i] != ids[i - 1]:
                collapsed.append(ids[i])

    final = [x for x in collapsed if x != blank_token]
    return final


import torch
from typing import List, Tuple, Dict, Union


class MDDEvaluator:
    def __init__(self):
        self.metrics_counts = {
            "TA": 0,
            "FA": 0,
            "TR": 0,
            "FR": 0,
            "CD": 0,
            "DE": 0
        }

        self.asr_counts = {
            "correct": 0,
            "substitution": 0,
            "deletion": 0,
            "insertion": 0,
            "total_canonical": 0
        }

    def _to_list(self, seq: Union[List[int], torch.Tensor]) -> List[int]:
        if isinstance(seq, torch.Tensor):
            return seq.detach().cpu().tolist()
        return list(seq)

    def calculate_edit_distance(self, hyp: Union[List[int], torch.Tensor], ref: Union[List[int], torch.Tensor]) -> \
    Tuple[int, int, int, int, List[Tuple[str, int]]]:

        hyp_list = self._to_list(hyp)
        ref_list = self._to_list(ref)

        n, m = len(hyp_list), len(ref_list)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if hyp_list[i - 1] == ref_list[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )

        i, j = n, m
        sub_cnt, del_cnt, ins_cnt, cor_cnt = 0, 0, 0, 0
        ref_mapping = [None] * m

        while i > 0 or j > 0:
            if i > 0 and j > 0 and hyp_list[i - 1] == ref_list[j - 1]:
                cor_cnt += 1
                ref_mapping[j - 1] = ("correct", hyp_list[i - 1])
                i -= 1
                j -= 1

            elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
                del_cnt += 1
                ref_mapping[j - 1] = ("del", -1)
                j -= 1

            elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
                ins_cnt += 1

                i -= 1

            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                sub_cnt += 1
                ref_mapping[j - 1] = ("sub", hyp_list[i - 1])
                i -= 1
                j -= 1

        return cor_cnt, sub_cnt, del_cnt, ins_cnt, ref_mapping

    def update_sample(self, canonical: Union[List[int], torch.Tensor],
                      transcript: Union[List[int], torch.Tensor],
                      output: Union[List[int], torch.Tensor]):
        canonical_list = self._to_list(canonical)

        _, _, _, _, trans_map = self.calculate_edit_distance(transcript, canonical)

        o_cor, o_sub, o_del, o_ins, out_map = self.calculate_edit_distance(output, canonical)

        self.asr_counts["correct"] += o_cor
        self.asr_counts["substitution"] += o_sub
        self.asr_counts["deletion"] += o_del
        self.asr_counts["insertion"] += o_ins
        self.asr_counts["total_canonical"] += len(canonical_list)

        for j in range(len(canonical_list)):
            trans_status, trans_val = trans_map[j]
            out_status, out_val = out_map[j]

            is_trans_correct = (trans_status == "correct")
            is_out_correct = (out_status == "correct")

            if is_trans_correct and is_out_correct:
                self.metrics_counts["TA"] += 1

            elif not is_trans_correct and is_out_correct:
                self.metrics_counts["FA"] += 1

            elif is_trans_correct and not is_out_correct:
                self.metrics_counts["FR"] += 1

            elif not is_trans_correct and not is_out_correct:
                self.metrics_counts["TR"] += 1

                if trans_status == out_status and trans_val == out_val:
                    self.metrics_counts["CD"] += 1
                else:
                    self.metrics_counts["DE"] += 1

    def compute_final_metrics(self) -> Dict[str, float]:
        mc = self.metrics_counts
        ac = self.asr_counts

        ta, fa, tr, fr = mc["TA"], mc["FA"], mc["TR"], mc["FR"]
        cd, de = mc["CD"], mc["DE"]

        total_detection = ta + fa + tr + fr
        det_accuracy = (ta + tr) / total_detection if total_detection > 0 else 0.0
        precision = tr / (tr + fr) if (tr + fr) > 0 else 0.0
        recall = tr / (tr + fa) if (tr + fa) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        frr = fr / (ta + fr) if (ta + fr) > 0 else 0.0
        far = fa / (tr + fa) if (tr + fa) > 0 else 0.0

        der = de / (cd + de) if (cd + de) > 0 else 0.0

        per = (ac["substitution"] + ac["deletion"] + ac["insertion"]) / ac["total_canonical"] if ac[
                                                                                                     "total_canonical"] > 0 else 0.0

        return {
            "Detection_Accuracy": det_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "FRR": frr,
            "FAR": far,
            "DER": der,
            "PER": per
        }


def evaluate(model, iterator, device, num_samples=3, idx2p=idx2P):
    model.eval()
    val_loss = 0
    evaluator = MDDEvaluator()
    result = []

    with torch.no_grad():
        for batch_idx, (waveform, transcript, canon, input_lengths, label_lengths, nccf) in enumerate(iterator):
            waveform, canon, nccf = waveform.to(device), canon.to(device), nccf.to(device)

            logits = model(waveform, canon, nccf)

            current_batch_size = logits.size(0)
            actual_time_steps = logits.size(1)

            ctc_logits = logits.transpose(0, 1)
            ctc_log_probs_for_loss = F.log_softmax(ctc_logits, dim=2)

            batch_log_probs = F.log_softmax(logits, dim=-1)

            CTC_BLANK_ID = 0
            PAD_ID = 3

            for i in range(current_batch_size):
                single_sample_probs = batch_log_probs[i]
                single_canon = canon[i].squeeze().cpu().tolist()
                single_trans = transcript[i].squeeze().cpu().tolist()

                phonemes, pred_ID = ctc_greedy_decode(single_sample_probs, idx2p=idx2p, blank_token=CTC_BLANK_ID)

                clean_output = [int(x) for x in pred_ID if int(x) != CTC_BLANK_ID]

                clean_canon = [int(x) for x in single_canon if int(x) != PAD_ID]
                clean_trans = [int(x) for x in single_trans if int(x) != PAD_ID]

                if CTC_BLANK_ID in clean_output:
                    print(f"Có token blank trong clean_prediction trong quá trình evaluate")

                evaluator.update_sample(
                    canonical=clean_canon,
                    transcript=clean_trans,
                    output=clean_output
                )
                metrics = evaluator.compute_final_metrics()
                result.append({
                    'canonical': str.join(" ", [idx2p[i] for i in clean_canon]),
                    'transcript': str.join(" ", [idx2p[i] for i in clean_trans]),
                    'prediction': str.join(" ", [idx2p[i] for i in clean_output])
                })
        return result, metrics


def ctc_greedy_decode(probs, idx2p, blank_token=0):
    best_path = torch.argmax(probs, dim=-1).cpu().numpy()
    collapsed_path = []
    last_token = None
    for token in best_path:
        if token != last_token:
            collapsed_path.append(token)
            last_token = token

    phonemes = [idx2p[token] for token in collapsed_path if token != blank_token]

    return phonemes, collapsed_path



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Decoder(vocab_size=len(P2idx), device=device)
model.to(device)

load_path = '/kaggle/input/datasets/nguyenquangduy15/papl-train-model/model.pth'
if os.path.exists(load_path):
    state_dict = torch.load(load_path, map_location=device)
    print("Đã load trọng số")
    model.load_state_dict(state_dict)
with open("mdd_dataset.pkl", "rb") as f:
    df = pickle.load(f)
result = []
result, metrics = evaluate(model, test_loader, device=device)
df['test']['predict'] = [item['prediction'] for item in result]
print(len(df['test']['canonical']))
print(len(df['test']['predict']))
up = pd.DataFrame(df['test'])
print(metrics)

up.to_csv("result.csv", index=False)
