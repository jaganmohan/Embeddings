from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inputgen import generate_batch
import tensorflow as tf

# Step 5: Begin training.

def train_model(session, model, reverse_dictionary, writer, data,
      batch_size, embedding_size, skip_window, num_skips, num_steps, log_dir):

  # We must initialize all variables before we use them.
  model.initialize()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window, data)

    feed_dict = {model.inputs: batch_inputs, model.labels: batch_labels}

    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [model._optimizer, model._merged, model._loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val

    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

  final_embeddings = model.norm_embeddings.eval()

  vocabulary_size = len(reverse_dictionary)
  print(vocabulary_size)
  # Write corresponding labels for the embeddings.
  with open(log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')
    
  return final_embeddings