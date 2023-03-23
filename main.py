import numpy as torch
import torch

import torch.nn as nn

import torch.linalg as L
import torch.functional as F

from nltk.corpus import brown
from nltk.probability import *
from nltk.collocations import *

from nltk import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer

def densityp(Pl:torch.tensor, Pr:torch.tensor):
    return torch.trace(Pl @ Pr) == 1

def phi(Pl:torch.tensor, M:torch.tensor, Pr:torch.tensor):
    return torch.trace(Pl@M@Pr@M.conj().transpose(1,0))

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
relative_frequency = frequency_distribution.freq("what")
relative_frequency

# get total ngrams
total_ngrams = len(frequency_distribution)
# create a random embedding space of it; embedding each in a 
embedding = nn.Embedding(total_ngrams, 8)
embedding(torch.tensor(0))


# questions
# 1. good corpus? brown is fine.
# 2. reliable generation scheme for M (dxd) than random?
# 3. m @ m.conj().transpose(1,0) definitely yields a semidefinite matrix, but
#    is it any good?

