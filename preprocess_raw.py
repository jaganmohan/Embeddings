import csv
import argparse
import collections
from src.preprocess import Preprocess

class PreprocessRawData(Preprocess):
    """ Modify this class according to the needs of custom data """
    
    def __init__(self):
        #super().__init__()
        pass
        
    def process_data(self):
         
        # Implement this method for custom data 
        # Example of pre-processing
        
        filename = "/data/jazzy/Embeddings/datasets/CIKM/train-item-views.csv"
        with open(filename) as f:
          reader = csv.reader(f, delimiter=";")
          sessions = collections.defaultdict(list)
          for i,r in enumerate(reader):
            if i == 0: continue;
            sessions[r[0]].append(r[2])

        filetxt = "data/productseq.txt"
        with open(filetxt, 'w') as f:
          for v in sessions.values():
            if len(v) > 1:
              f.write('%s\n' %(' '.join(v)))
        
                    
def main(args):
    print("Starting pre-processing...")
    PreprocessRawData().process_data()
    print("Finished processing")
    
def parseargs():
    parser = argparse.ArgumentParser()
    FLAGS,unparsed=parser.parse_known_args()
    return FLAGS 
    
if __name__ == "__main__":
    args = parseargs()
    main(args)
