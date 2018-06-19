import argparse
import csv
import os
import numpy as np
import pandas as pd
import linecache

def main(args):
    
    # Load labels 
    labels = {}
    r_labels = {}
    with open(args.labels_file_path) as f:
        
        reader = csv.reader(f, delimiter='\t')
        for i,label in enumerate(reader):
            labels[label[0]] = i
            r_labels[i] = label[0]
            
        # Load similarities
        with open(args.sim_file_path) as f:
            for i, line in enumerate(f):
                if i==labels[args.word]:
                    print(i)
                    sim = line
                    break
        print("Loaded similarities")
        
        sim = [-float(i) for i in sim.split(' ')]
            
        top_k = 8  # number of nearest neighbors
        nearest = np.array(sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % args.word
        for k in xrange(top_k):
            close_word = r_labels[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
        
def parseargs():
    current_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('word', type=str, help='Word to find synonyms of')
    parser.add_argument('--sim_file_path', type=str,
                       default=os.path.join(current_path, 'similarity.txt'),
                       help='Path to vocabulary similarity file')
    parser.add_argument('--labels_file_path', type=str,
                       default=os.path.join(current_path,'log','metadata.tsv'),
                       help='Path to vocabulary labels file')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS
    
if __name__ == '__main__':
    args = parseargs()
    main(args)