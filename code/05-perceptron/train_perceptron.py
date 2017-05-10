#!/usr/bin/env python
# coding: utf-8
#
# Filename:   train_perceptron.py
# Author:     Peinan ZHANG
# Created at: 2017-05-10

import sys
from collections import defaultdict
import pickle
import numpy as np


N_EPOCH = 10
WORD_COUNT = defaultdict(int) # weight

def load_data(data_fp):
  lines = [ line.rstrip('\n') for line in open(data_fp) ]

  return lines


def parse_line(line):
  y, sentence = line.split('\t')
  y = int(y)
  x = defaultdict(int)
  for word in sentence.split(' '):
    x[word] += 1

  return (y, x)


def predict_one(x, W):
  """ y_ = sign(sigma(Wx)) """
  Wx = sum([ W[word] * x[word] for word, count in x.items() ])
  y_ = sign(Wx)

  return y_


def sign(x):
  f_x = 1
  if x < 0:
    f_x = -1

  return f_x


def dump_model(model_fp):
  with open(model_fp, 'w') as out:
    for word, count in WORD_COUNT.items():
      out.write("{}\t{}\n".format(word, count))

  print("Model file is dumped to {}".format(model_fp))


def main():
  data_fp  = sys.argv[1]
  model_fp = sys.argv[2]

  lines = load_data(data_fp)
  for epoch in range(N_EPOCH):
    print("[EPOCH] {:3d}".format(epoch+1))
    for line in lines:
      y, x = parse_line(line)
      y_ = predict_one(x, WORD_COUNT)
      is_correct = y_ == y
      print(line, y_, y)
      if not is_correct:
        if y_ == 1:
          for word, _ in x.items():
            WORD_COUNT[word] += -1
        elif y_ == -1:
          for word, _ in x.items():
            WORD_COUNT[word] += 1

  print("Training is done.")
  dump_model(model_fp)


if __name__ == '__main__':
  main()
