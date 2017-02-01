#!/usr/bin/env python3
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2017-01-27

import sys, pickle as pkl
from collections import defaultdict

def load_train_file(filepath):
  word_count_map   = defaultdict(int)
  total_word_count = 0
  for line in open(filepath):
    line = line.rstrip('\n') + ' </s>'
    for word in line.split(' '):
      word_count_map[word] += 1
      total_word_count += 1

  return word_count_map, total_word_count


def make_unigram_lm(word_count_map, total_word_count):
  """Make an unigram language model
  P(w_i) = word_count / total_word_count
  """
  unigram_lm = {}
  for word, word_count in word_count_map.items():
    unigram_lm[word] = word_count / float(total_word_count)

  return unigram_lm


def write_lm(lm, filepath, mode='plain'):
  if mode == 'plain':
    with open(filepath, 'w') as outfile:
      for word, p in lm.items():
        outfile.write("%s\t%.10f\n" % (word, p))
  elif mode == 'pickle':
    pkl.dump(lm, open(filepath, 'w'))

  print("Successfully write language model to '%s'." % filepath)


def print_dict(target_dict, sort=True, reverse=True, max_item=10):
  items = target_dict.items()
  len_keys = len(target_dict.keys())
  if sort:
    items = sorted(target_dict.items(), key=lambda x:x[1], reverse=reverse)
  if not max_item:
    max_item = len_keys
  items = items[:max_item]

  for k, v in items:
    print(k, v)
  print("Total key count:", len_keys)


def main():
  word_count_map, total_word_count = load_train_file(sys.argv[1])
  lm = make_unigram_lm(word_count_map, total_word_count)

  print("Word count map")
  print_dict(word_count_map)
  print("Total word count:", total_word_count)
  print("Language model")
  print_dict(lm)

  write_lm(lm, sys.argv[2])


if __name__ == '__main__':
  main()
