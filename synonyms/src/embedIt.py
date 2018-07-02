from flask import Flask, request
from src.synonyms import Synonyms

syn = None
config = {}
app = Flask(__name__)
app.config.from_envvar('APP_CONFIG')

def word_in_data(word):
  if(word in syn.items2ID().keys()):
    return True
  else:
    return False

@app.route("/search/<string:word>")
def search(word):
  if word_in_data(word):
    return word
  else:
    return "Item not found in processed data"

@app.route("/allItems")
def get_items():
  return str(syn.items2ID().keys())

@app.route("/embeddings")
def get_embeddings():
  return syn.items

@app.route("/similar", methods=['GET'])
def top_similar():
  word=request.args.get('word')
  k=request.args.get('k')
  if k == None: k=10
  else: k = int(k)
  if word != None and word_in_data(word):
    return "Nearest to {}:{}".format(word,syn.n_neighbors(word,k))
  else:
    return "Item:{} not found in processed data".format(word)

if __name__ == "__main__":
  #print(app.config)
  syn = Synonyms(app.config['EMB_FILE'],app.config['LABELS_FILE'])
  app.run(host=app.config['IP'], 
            port=int(app.config['PORT']))

