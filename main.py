import numpy as torch
import torch

import torch.nn as nn
import torch.nn.utils as utils

import torch.linalg as L
import torch.functional as F

from nltk.corpus import brown
from nltk.probability import *
from nltk.collocations import *

from nltk import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer

class SemidefiniteMap(nn.Module):
    """A positive semidefinite operator like Pl, Pr

    Parameters
    ----------
    size : int
        the in/out dimension of the operator
    """

    def __init__(self, size):
        super().__init__()

        self.U = utils.parametrizations.orthogonal(nn.Linear(size, size))
        self.V = utils.parametrizations.orthogonal(nn.Linear(size, size))
        self.singular_values_unrelu = nn.Parameter(torch.rand(size))
        self.relu = nn.ReLU()

    def get(self):
        singular_values = self.relu(self.singular_values_unrelu)
        return self.U.weight @ torch.diag(singular_values) @ self.V.weight.T

    def forward(self,x):
        singular_values = self.relu(self.singular_values_unrelu)

        return self.get() @ x

def phi(Pl:torch.tensor, M:torch.tensor, Pr:torch.tensor):
    return torch.trace(Pl@M@Pr@M.conj().transpose(1,0))

class QNLPModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.D = nn.Linear(1, hidden_dim)

        self.Pl = SemidefiniteMap(hidden_dim)
        self.Pr = SemidefiniteMap(hidden_dim)

    def forward(self, x):
        embedded = self.D(self.embedding(x).unsqueeze(2))
        M_dxd = torch.matmul(*embedded)
        phi_M = phi(self.Pl.get(), M_dxd, self.Pr.get())

        return phi_M
        
model = QNLPModel(8, 4)
model(torch.tensor([2,3]))



def densityp(Pl:torch.tensor, Pr:torch.tensor):
    return torch.trace(Pl @ Pr) == 1

def semidefinitify(m:torch.tensor):
    """take m and appling to its conjugate transpose, hence returning a semidefinte tensor

    Conditions for semidefinity: z* M z is real positive for any nonzero complex vector z

    Small proof for why this works:
    z* M M* z = (M* z)* (M* z); and so the inner product between them will cause negative
      by negative, imag to imag, making it real and positive

    Question for ted: wouldn't this make all M real? Isn't that a problem?
    """
    return m @ m.conj().transpose(1,0)




# prepare n-gram frequency data
vect = CountVectorizer(analyzer='word', ngram_range=(1,3))
analyzer = vect.build_analyzer()
results = analyzer(' '.join(brown.words()))
results.reverse() # because we want the LEAST frequent first
frequency_distribution = nltk.FreqDist(results)

# and now
frequency_distribution.freq("what is")
relative_frequency

# get total ngrams
total_ngrams = len(frequency_distribution)
total_ngrams
# create a random embedding space of it; embedding each in a 
embedding = nn.Embedding(total_ngrams, 8)
embedding(torch.tensor(4))


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
