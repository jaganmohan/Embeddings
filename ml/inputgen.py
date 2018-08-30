import numpy as np
import collections
import random

# Step 3: Function to generate a training batch for the skip-gram model.

data_index = 0

def generate_batch(batch_size, num_skips, skip_window, data):
  """This method generates batches over training data"""
  
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  
  # variable declarations with span window over data
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  
  # resetting data_index at the end
  if data_index + span > len(data):
    data_index = 0
  # storing context in buffer  
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  
  for i in range(batch_size // num_skips):
    # context word indices
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  
  return batch, labels
