import pickle
import json
import os
import torch
from sklearn.model_selection import train_test_split
import torchaudio
import pandas as pd


with open ("/kaggle/working/mdd_dataset.pkl","rb") as f:
    df = pickle.load(f)

with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)
with open("/kaggle/working/lexicon.pkl",'rb') as f:
    lexicon = pickle.load(f)


p2idx = vocab['p2idx']


def encode_sequence(text_sequence, p2idx=p2idx):
    if pd.isna(text_sequence) or not text_sequence:
        return []

    sequence_ids = []
    for i in text_sequence.split():
        if i not in p2idx:
            sequence_ids.append("<unk>")
        else:
            sequence_ids.append(p2idx[i])

    return sequence_ids

def encode_data(data):
    encoded_samples = []

    for path,canon,trans in zip(data['path'],data['canonical'],data['transcript']):

        canon_seq = encode_sequence(canon, p2idx)
        trans_seq = encode_sequence(trans, p2idx)

        encoded_samples.append(
            {
                "path": path,
                "canonical": torch.tensor(canon_seq, dtype=torch.long),
                "transcript": torch.tensor(trans_seq, dtype=torch.long),
            }
        )

    return encoded_samples

train_samples = encode_data(df['train'])
dev_samples = encode_data(df['dev'])


dataset_dict = {
    'train': train_samples,
    'dev' : dev_samples
}

with open('Encoded.pkl', 'wb') as f:
    pickle.dump(dataset_dict, f)