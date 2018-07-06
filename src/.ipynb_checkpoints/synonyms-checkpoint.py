from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import numpy as np

class Synonyms:
    
    def __init__(self, emb_file, labels_file):
        
        print("""Initializing Synonyms object with:\n
        embeddings file: %s\n
        labels file:%s"""%(emb_file, labels_file))
        
        # Load embeddings
        self.embeddings = np.loadtxt(emb_file)
         
        self.labels = {}
        self.r_labels = {}
        self._vocab={'items':[]}
        # Load labels
        with open(labels_file) as f:
            reader = csv.reader(f)
            for i,pid in enumerate(reader):
                self.labels[pid[0]] = i
                self.r_labels[i] = pid[0]
                self._vocab['items'].append({
                    'label':pid[0],
                    'emb':self.embeddings[i]
                })
                
        #del embeddings
                
    @property
    def items(self):
        return json.dumps(self._vocab)
                
    def items2ID(self):
        return self.labels
    
    def id2items(self):
        return self.r_labels
        
    #def train(self):   
        
    def n_neighbors(self, word, k=10):
        item_emb = self.embeddings[self.labels[word]]
        sim = np.matmul(self.embeddings, item_emb)
        sim *= -1
        nearest = np.array(sim).argsort()[1:k + 1]
        res = []
        for i in range(k):
            res.append(r_labels[nearest[i]])
        return res