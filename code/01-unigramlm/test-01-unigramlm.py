#!/usr/bin/env python3
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2017-01-27

import sys, pickle as pkl, math
from collections import defaultdict

def load_unigram_lm(filepath, mode='plain'):
  unigram_lm = {}
  if mode == 'plain':
    for line in open(filepath):
      line = line.rstrip('\n')
      word, prob = line.split('\t')
      unigram_lm[word] = float(prob)
  elif mode == 'pickle':
    unigram_lm = pkl.load(open(filepath))

  return unigram_lm


def eval_model(filepath, lm, lambda_unk=.05, N=100000):
  """Evaluate the test dataset. This function returns entropy and coverage.

  Probability:
    P(w_i) = (1 - lambda_unk) * P_ML(w_i) + lambda_unk * 1/N
  Log Likelihood:
    log P(W_test | M) = sigma ( log P(w | M) )
  Entropy:
    H(W_test | M) = 1/|W_test| sigma ( -log2 P(w | M) )
  Perplexity:
    PPL = 2^H
  Coverage:
    The rate of the words appeared in the test dataset that are covered in the 
    language model.
  """

  total_word_count = 0
  not_found_word_map = defaultdict(int)
  H = 0
  for line in open(filepath):
    line = line.rstrip('\n') + ' </s>'
    for word in line.split(' '):
      try:
        P = (1 - lambda_unk) * lm[word] + lambda_unk / float(N)
      except KeyError:
        P = lambda_unk / float(N)
        not_found_word_map[word] += 1
      total_word_count += 1
      H -= math.log(P, 2)
  
  H = H/float(total_word_count)
  lm_keys = len(lm.keys())
  coverage = (lm_keys - len(not_found_word_map.keys())) / float(lm_keys)

  return H, coverage


def main():
  lm = load_unigram_lm(sys.argv[1])
  H, coverage = eval_model(sys.argv[2], lm)
  print("Entropy: %f, Coverage: %s" % (H, coverage))


if __name__ == '__main__':
  main()
