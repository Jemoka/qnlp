import numpy as torch
import torch
import tqdm

import torch.nn as nn
import torch.nn.utils as utils
from torch.optim import AdamW, SGD

import torch.linalg as L
import torch.functional as F

from nltk.corpus import brown
from nltk.probability import *
from nltk.collocations import *

from nltk import sent_tokenize

import nltk 
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

# TODO
# - search DIRECTLY on these matricies for SemidefiniteDensityMap 
# - missing: COMPLEX NUMBERS!!!! AAAAAAAAA; use polar form
class SemidefiniteDensityMap(nn.Module):
    """A positive semidefinite operator like Pl, Pr

    Parameters
    ----------
    size : int
        the in/out dimension of the operator
    """

    def __init__(self, size):
        super().__init__()

        self.U = utils.parametrizations.orthogonal(nn.Linear(size, size).to(torch.cfloat))
        self.V = utils.parametrizations.orthogonal(nn.Linear(size, size).to(torch.cfloat))
        self.singular_values = nn.Parameter(torch.rand(size))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def get(self):
        return self.U.weight @ torch.diag(self.relu(self.singular_values)).to(torch.cfloat) @ self.V.weight.T

    def forward(self,x):
        singular_values = self.relu(self.singular_values)

        return self.get() @ x

m = SemidefiniteDensityMap(2)

def phi(Pl:torch.tensor, M:torch.tensor, Pr:torch.tensor):
    return torch.trace(Pl@M@Pr@M.conj().transpose(1,0))

class QNLPModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.hidden = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.D = nn.Linear(1, hidden_dim).to(torch.cfloat)

        self.Pl = SemidefiniteDensityMap(hidden_dim)
        self.Pr = SemidefiniteDensityMap(hidden_dim)

    def forward(self, x):
        embedded = self.D(self.embedding(x).unsqueeze(2).to(torch.cfloat))
        base_matrix = torch.eye(self.hidden).to(torch.cfloat)
        for i in reversed(embedded):
            base_matrix = i@base_matrix
        phi_M = phi(self.Pl.get(), base_matrix, self.Pr.get())

        return phi_M

    def get_m(self, x):
        embedded = self.D(self.embedding(x).unsqueeze(2))
        base_matrix = torch.eye(self.hidden)
        for i in reversed(embedded):
            base_matrix = i@base_matrix
        return base_matrix

# prepare n-gram frequency data
vect = CountVectorizer(analyzer='word', ngram_range=(1,3))
analyzer = vect.build_analyzer()
results = analyzer(' '.join(brown.words()))
results.reverse() # because we want the LEAST frequent first
frequency_distribution = nltk.FreqDist(results)

# build vocab list
vocab = set(brown.words())
vocab_ids = defaultdict(lambda:len(vocab_ids))
vocab_id_reverse = {vocab_ids[word]:word for word in vocab}
vocab_ids = dict(vocab_ids)

# model 
model = QNLPModel(len(vocab_ids), 256)
optim = AdamW(model.parameters(), lr=3e-5)

# for each group, train
bar = tqdm.tqdm(frequency_distribution.elements())
for group in bar:
    freq = torch.tensor(frequency_distribution.freq(group))
    words = group.split(" ")
    try:
        in_tensor = torch.tensor([vocab_ids[i] for i in words])
    except KeyError:
        # OOV
        optim.zero_grad()
        continue
    out_tensor = model(in_tensor)
    mse = (freq - out_tensor)**2
    mse.backward()
    optim.step()
    optim.zero_grad()

    bar.set_description(f"loss: {round(torch.norm(mse).item(), 3)}")





# model.embedding(torch.tensor(vocab_ids[]))



# model
# brown.words()

# # get total ngrams
# total_ngrams = len(frequency_distribution)
# total_ngrams
# # create a random embedding space of it; embedding each in a 
# embedding = nn.Embedding(total_ngrams, 8)
# embedding(torch.tensor(4))


# questions
# 1. good corpus? brown is fine. <-
# 2. reliable generation scheme for M (dxd) than random? <-
# 3    => outer product + slight pertubation (add uniform random vector) 
# 3. m @ m.conj().transpose(1,0) definitely yields a semidefinite matrix, but
#    is it any good?
#     => NO. more efficiently: start with eigenvalues + multiply by random orthogonal
#            vectors.
#
#        if each entries of nxn is IID distributed, what is its distribution of eigenvals?
#        don't start with uniformly random eigenvals, instead, control it: i.e. as M(nxn) => n goes
#        to infinity, GUI: gaussian [?] [?]
#
#    degrees of freedom: 1) eigenvalues; theorems above matches 2) direction of the orthonorm bases
# 4. where to start M?? "Hierarchical optimization"
#    split up decision variables into 2 parts
#
# 5. losopolous: language as a matrix product state
#     HIGHARCHICAL OPTIMIZATION: master coordinates solutions of slave problems
#     master: Solivng Pl, Pr; slave: optimal M given Pl. Pr
#     Actor-Critic 
#
# non-nergotic change
