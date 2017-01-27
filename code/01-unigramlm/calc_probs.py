#!/usr/bin/env python
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2017-01-27

import sys, math

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


def calc_unigram_prob(lm, text, lambda_unk=.05, N=1e+6):
  P_text = 0
  for word in text.split(' '):
    try:
      P = (1 - lambda_unk) * lm[word] + lambda_unk / float(N)
    except KeyError:
      P = lambda_unk / float(N)
    P_text += math.log(P)

  return P_text


def main():
  lm = load_unigram_lm(sys.argv[1])
  for line in sys.stdin.readlines():
    print(calc_unigram_prob(lm, line))


if __name__ == '__main__':
  main()
