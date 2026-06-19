import pandas as pd
import pickle
import torch
import os
import re


def load_lexicon(lexicon_path):
    lexicon_dict = {}
    all_phoneme = []
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            word = parts[0].lower()
            phonemes = parts[1:]

            lexicon_dict[word] = str.join(" ", phonemes)
    return lexicon_dict


lexicon = load_lexicon(
    "/kaggle/input/datasets/nguyenquangduy15/challenge-train/MDD-Challenge-2025-training-set/metadata/lexicon_vmd.txt")
r_lex = {phone: word for word, phone in lexicon.items()}
dictionary = {'w2p': lexicon,
              'p2w': r_lex}
with open("lexicon.pkl", "wb") as f:
    pickle.dump(dictionary, f)

word = pd.read_csv(
    "/kaggle/input/datasets/nguyenquangduy15/challenge-train/MDD-Challenge-2025-training-set/metadata/train.csv")
root_dir = "/kaggle/input/datasets/nguyenquangduy15/challenge-train/MDD-Challenge-2025-training-set/"


def clean_and_split(text):
    if not isinstance(text, str):
        return []
    text = text.lower().strip()
    text = re.sub(r'[.,?!:;()\"\'\-]', '', text)
    return text.split()


def s2p(list, lexicon=lexicon):
    return str.join(" ", [lexicon[i] for i in list])


def convert_to_phoneme(lexicon=lexicon, csv=word, root_dir=root_dir):
    meta = []

    for path, canonical, transcript in zip(csv['path'], csv['canonical'], csv['transcript']):
        abspath = os.path.join(root_dir, path)
        canon_words = clean_and_split(canonical)
        trans_words = clean_and_split(transcript)

        canon = s2p(canon_words)

        trans = s2p(trans_words)

        meta.append({
            'path': abspath,
            'canonical': canon,
            'transcript': trans
        })
    return meta


conversion = convert_to_phoneme(lexicon, word)

train_subset, dev_subset = torch.utils.data.random_split(conversion, [0.8, 0.2])

train_list = [conversion[i] for i in train_subset.indices]
dev_list = [conversion[i] for i in dev_subset.indices]
train_df = pd.DataFrame(train_list)
dev_df = pd.DataFrame(dev_list)

train_dev = {
    'train': train_df,
    'dev': dev_df
}
with open("mdd_dataset.pkl", "wb") as f:
    pickle.dump(train_dev, f)

print("Đã tạo xong metadata")
