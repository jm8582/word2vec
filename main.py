import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from preprocess import Preprocessor

import torch
from torch.utils.data import DataLoader, Dataset

negative_sampling = False
EMBEDDING_DIM = 2
WINDOW_SIZE = 2
DSET_NAME = 'synthetic'
BATCH_SIZE = 1
MAX_EPOCH = 1000
GPU = 0


class Word2Vec(torch.nn.Module):
    def __init__(self, word_dim, embed_dim):
        super(Word2Vec, self).__init__()
        self.word_dim = word_dim
        self.embed_dim = embed_dim

        self.enc = torch.nn.Embedding(word_dim, embed_dim)
        self.dec = torch.nn.Linear(embed_dim, word_dim)
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class NewDataSet(Dataset):
    def __init__(self, dset):
        self.dset = dset
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.dset[idx]


preprocessor = Preprocessor(dset_name=DSET_NAME)
preprocessor.preprocess()
device = torch.device(f"cuda:{GPU}")

net = Word2Vec(preprocessor.n_corpus_words, EMBEDDING_DIM)
net.to(device)
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)

dset = []
for sentence in preprocessor.corpus_num:
    for c, center in enumerate(sentence):
        for context in sentence[max(0, c-WINDOW_SIZE):c+WINDOW_SIZE+1]:
            if context != center:
                dset.append((center, context))

dataset = NewDataSet(dset)
dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(MAX_EPOCH):
    tot_loss = 0
    for center, context in dl:
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad()
        out = net(center)

        loss = F.cross_entropy(out, context)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        print(f"{epoch=}, {tot_loss=:.2f}")

embeddings = net.enc.weight.detach().to('cpu').numpy()
x = embeddings.T[0]
y = embeddings.T[1]
fig, ax = plt.subplots(dpi=300)
ax.scatter(x, y)
for i, txt in enumerate(preprocessor.corpus_words):
    ax.annotate(txt, (x[i], y[i]))
fig.savefig('embedding_space2.png', dpi=300)