#!/usr/bin/env python

# Apache 2.0

import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', '-v', dest='exclude',
                        action='store_true', help='exclude filter words')
    parser.add_argument('filt', type=str, help='filter list')
    parser.add_argument('infile', type=str, help='input file')
    args = parser.parse_args()

    vocab = set()
    with open(args.filt) as vocabfile:
        for line in vocabfile:
            vocab.add(line.strip())

    with open(args.infile) as textfile:
        for line in textfile:
            if args.exclude:
                print(" ".join(
                    map(lambda word: word if not word in vocab else '', line.strip().split())))
            # else:
            #     print(" ".join(map(lambda word: word if word in vocab else '<UNK>', unicode(line, 'utf_8').strip().split())).encode('utf_8'))
