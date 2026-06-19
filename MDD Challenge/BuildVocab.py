import pickle
import pandas as pd

with open ("lexicon.pkl",'rb') as f:
    df = pickle.load(f)

w2p = df['w2p']
vocab = ["", "<eos>", "<sos>", "<pad>", "<unk>"]
for word, phoneme in w2p.items():
    for i in phoneme.split():
        if i not in vocab:
            vocab.append(i)

idx2p = {i: phoneme for i, phoneme in enumerate(vocab)}
p2idx = {phoneme: i for i, phoneme in enumerate(vocab)}

assert isinstance(idx2p, dict), "idx2p không phải là dict!"

with open ('vocab.pkl','wb') as f:
    pickle.dump({'idx2p':idx2p,'p2idx':p2idx},f)