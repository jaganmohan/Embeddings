from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

# pylint: disable=missing-docstring
# pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf
import math
import os
import sys
import argparse
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.tensorboard.plugins import projector

from preprocess import build_dataset, read_data
from train import train_model
# Step 4: Build and train a skip-gram model.

class Word2Vec():
    
    def __init__(self, batch_size, embedding_size, vocabulary_size,
                num_sampled):
        
      self.graph = tf.Graph()
      with self.graph.as_default():
        # Input data.
        with tf.name_scope('inputs'):
          self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
          self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
          # Look up embeddings for inputs.
          with tf.name_scope('embeddings'):
            self._embeddings = tf.Variable(
              tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self._embeddings, self.train_inputs)

          # Construct the variables for the NCE loss
          with tf.name_scope('weights'):
            nce_weights = tf.Variable(
              tf.truncated_normal(
                  [vocabulary_size, embedding_size],
                  stddev=1.0 / math.sqrt(embedding_size)))
          with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
          self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', self.loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
          self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        # Compute the cosine similarity between all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self._embeddings), 1, keepdims=True))
        self.normalized_embeddings = self._embeddings / norm
        self._similarity = tf.matmul(
          self.normalized_embeddings, self.normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        self.merged = tf.summary.merge_all()

        # Add variable initializer.
        self.init = tf.global_variables_initializer()
        
        # Create a saver.
        self._saver = tf.train.Saver()
    
    @property
    def _graph(self):
        return self.graph
    
    @property
    def inputs(self):
        return self.train_inputs
    
    @property
    def labels(self):
        return self.train_labels
    
    @property
    def _optimizer(self):
        return self.optimizer
        
    @property
    def _merged(self):
        return self.merged
    
    @property
    def _loss(self):
        return self.loss
    
    @property
    def similarity(self):
        return self._similarity
    
    @property
    def embeddings(self):
        return self._embeddings
    
    @property
    def norm_embeddings(self):
        return self.normalized_embeddings
    
    @property
    def savemodel(self):
        return self._saver
    
    def initialize(self):
        self.init.run()

# Step 6: Visualize the embeddings.
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

def runTSNE(embeddings, reverse_dictionary):
    try:
      tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
      plot_only = 500
      low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels, os.path.join(os.getcwd(), 'word2vec.png'))

    except ImportError as ex:
      print('Please install sklearn, matplotlib, and scipy to show embeddings.')
      print(ex)
    
def main(current_path, args):
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    filename = args.file
    
    # preprocessing raw data and building vocabulary
    vocabulary = read_data(filename)
    print('Data size', len(vocabulary))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000

    # Filling 4 global variables:
    # data - list of codes (integers from 0 to vocabulary_size-1).
    # This is the original text but words are replaced by their codes
    # count - map of words(strings) to count of occurrences
    # dictionary - map of words(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to words(strings)
    data, count, dictionary, reverse_dictionary = build_dataset(
      vocabulary, vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    
    num_sampled = 64  # Number of negative examples to sample.
    train_args = {
      'batch_size' : 128,
      'embedding_size' : 128,  # Dimension of the embedding vector.
      'skip_window' : 1,  # How many words to consider left and right.
      'num_skips' : 2,  # How many times to reuse an input to generate a label.
      'num_steps' : 100001,  # Number of epochs
      'log_dir' : args.log_dir
    }
    
    model = Word2Vec(128, 128, vocabulary_size,
                    num_sampled)
    graph = model._graph
    embeddings = None
    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(args.log_dir, session.graph)
        embeddings = train_model(session, model, 
                          reverse_dictionary, writer, data,**train_args)
        similarity = model.similarity.eval()
        np.savetxt("similarity.txt", similarity)
        
        # Save the model for checkpoints.
        model.savemodel.save(session, os.path.join(args.log_dir, 'model.ckpt'))
        
        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = model.embeddings.name
        embedding_conf.metadata_path = os.path.join(args.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)
      
        
    # Closing writer
    writer.close()
    
    runTSNE(embeddings, reverse_dictionary)

def parseargs(current_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
        default=os.path.join(current_path, 'data', 'text8.zip'))
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path,'log'),
        help='The log directory for TensorBoard summaries.')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

if __name__ == "__main__":
    current_path = os.getcwd()
    args = parseargs(current_path)
    main(current_path, args)
    