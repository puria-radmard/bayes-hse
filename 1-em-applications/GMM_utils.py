import numpy as np
from numpy.linalg import slogdet, det, solve
import matplotlib.pyplot as plt

import math
from math import sqrt, e
from math import pi as mathpi

def generate_spd_sigma(C, d):

  sigmas = []

  for c in range(C):

    det = -1

    while det <= 0:
      sigma = np.random.rand(d, d)
      # Make sigma symmetric:
      sigma = (sigma + sigma.T) / 2
      det = np.linalg.det(sigma)
    sigmas.append(sigma)
  
  sigma = np.stack(sigmas, axis = 0)
  
  return sigma


def M_step_mus(X, gamma):

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    class_mus = []

    for c in range(C):
      numerator = gamma[0][c] * X[0]
      denominator = gamma[0][c]
      for i in range(1, N):
        numerator += gamma[i][c] * X[i]
        denominator += gamma[i][c]
      class_mu = numerator / denominator
      assert len(class_mu) == d
      class_mus.append(class_mu)

    mu = np.stack(class_mus, axis = 0)

    return mu


def M_step_sigma(X, gamma, mu):

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    class_sigmas = []

    for c in range(C):
      diff = X[0] - mu[c]
      numerator = np.outer(diff, diff) * gamma[0][c]
      denominator = gamma[0][c]
      for i in range(1, N):
        diff = X[i] - mu[c]
        numerator += np.outer(diff, diff) * gamma[i][c]
        denominator += gamma[i][c]
      class_sigma = numerator / denominator
      class_sigmas.append(class_sigma)

    sigma = np.stack(class_sigmas, axis=0)
    
    return sigma


def M_step_pi(X, gamma):

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    pi = []

    for c in range(C):
      Nc = np.sum(gamma[:,c])
      pi.append(Nc / N)
    
    return np.array(pi)

def f_gaussian(X, mu, sigma):

  n = len(mu)
  assert len(sigma) == n
  assert len(sigma[0]) == n
  assert len(X) == n

  norm_const = 1 / (((2 * mathpi)**(n / 2)) * (np.linalg.det(sigma)**(0.5)))
  diff = X - mu
  exponent = -0.5 * np.matmul(diff.T, np.linalg.solve(a = sigma, b = diff))

  p = norm_const * (e**exponent)

  if np.isnan(p):
    import pdb; pdb.set_trace()

  return p

def softmax(probs):
  
  max_p = np.array([max(probs) for p in probs])
  norm_const = 1 / sum(e**(probs - max_p))

  smp = norm_const * e**(probs - max_p) 
  return smp

def normalise_probs(probs):
  norm_const = 1 / sum(probs)
  norm_p = norm_const * probs

  return norm_p
