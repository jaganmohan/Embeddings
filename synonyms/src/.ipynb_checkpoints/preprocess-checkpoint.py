import zipfile
import tensorflow as tf
import collections

# Read the data into a list of strings.
def read_data(filename, isZip=False):
  if isZip:  
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  else:
    with open(filename) as f:
      data = tf.compat.as_str(f.read()).split()
  return data

def build_dataset(words, n_words=0):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common())#n_words - 1))
  #count = [['UNK', -1]]
  #tmp = filter(lambda x:x[1]>n_words, count)
  #for i in tmp:
  #  count.append(i)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
