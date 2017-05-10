#!/usr/bin/env python
# coding: utf-8
#
# Filename:   test_perceptrion.py
# Author:     Peinan ZHANG
# Created at: 2017-05-10

import sys
from collections import defaultdict


def load_data(data_fp):
  lines = [ line.rstrip('\n') for line in open(data_fp) ]

  return lines


def load_model(model_fp):
  model = {}
  for line in open(model_fp):
    word, weight = line.rstrip('\n').split('\t')
    model[word] = int(weight)

  return model


def parse_line(line):
  y, sentence = line.split('\t')
  y = int(y)
  x = defaultdict(int)
  for word in sentence.split(' '):
    x[word] += 1

  return (y, x)


def predict_one(x, W):
  """ y_ = sign(sigma(Wx)) """
  Wx = sum([ W[word] * x[word] for word, count in x.items() if word in W])
  y_ = sign(Wx)

  return y_


def sign(x):
  f_x = 1
  if x < 0:
    f_x = -1

  return f_x


def main():
  data_fp = sys.argv[1]
  model_fp = sys.argv[2]

  lines = load_data(data_fp)
  model = load_model(model_fp)
  total_count = 0
  correct_count = 0
  for line in lines:
    y, x = parse_line(line)
    y_ = predict_one(x, model)
    is_correct = y_==y
    print("[{}] {}\nPRED:{}, TRUE:{}".format(is_correct, line, y_, y))

    total_count += 1
    if is_correct:
      correct_count += 1

  print("\nAccuracy: {:.03f} % ({} / {})"
      .format(100 * float(correct_count) / total_count, correct_count, total_count))


if __name__ == '__main__':
  main()
