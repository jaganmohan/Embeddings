import os
import numpy as np
import csv
import collections

filename = "/data/jazzy/Embeddings/datasets/CIKM/train-item-views.csv"
with open(filename) as f:
    reader = csv.reader(f, delimiter=";")
    sessions = collections.defaultdict(list)
    for i,r in enumerate(reader):
        if i == 0: continue;
        sessions[r[0]].append(r[2])

filetxt = "../data/productseq.txt"
with open(filetxt, 'w') as f:
    for v in sessions.values():
        if len(v) > 1:
            f.write('%s\n' %(' '.join(v)))