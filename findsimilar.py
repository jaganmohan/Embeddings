"""
Custom script to find similar items in the data
using provided API in Synonyms class
"""

import csv
import os
import argparse
import numpy as np

from src.synonyms import Synonyms

def main(args):
    
    print("Using parameters:\n",args)
    
    syn = Synonyms(args.emb_file, args.labels_file)
    r_labels = syn.id2items()
    prod_categories = "/data/jazzy/Embeddings/datasets/CIKM/product-categories.csv"
    prod_cat = {}
    with open(prod_categories) as cat:
        reader = csv.reader(cat, delimiter=";")
        for pid, cat in reader:
            prod_cat[pid] = cat
    
    nearest = syn.n_neighbors(args.word)
    print('Nearest to %s:' % (args.word))
    print('Items:',nearest)
    """print('Nearest to %s(%s):' % (args.word,prod_cat[args.word]))
    print('Item_ID \t Category_ID')
    for k in xrange(args.n_num):
        close_word = r_labels[nearest[k]]
        print('%s \t\t %s' % (close_word, prod_cat[close_word]))"""
        
def parseargs():
    current_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('word', type=str, help='Word to find synonyms of')
    parser.add_argument('n_num', type=int, default=10,
                       help='Top k similar items')
    parser.add_argument('emb-file', type=str,
                       default=os.path.join(current_path,'output','embeddings.txt'),
                       help='Path to vocabulary similarity file')
    parser.add_argument('labels-file', type=str,
                       default=os.path.join(current_path,'log','metadata.tsv'),
                       help='Path to vocabulary labels file')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS
    
if __name__ == '__main__':
    args = parseargs()
    main(args)
