import numpy as np
from random import shuffle
import sys

class Cache:
    """
    Originally, transforming each piece into a state matrix for input and output
    took nearly 2 seconds for each batch. Since our training data and model do not take
    that much space, we can instead simply store the transformed inputs and outputs
    in memory.
    """

    def __init__(self, max_pieces=0):
        self.input_storage = {}
        self.output_storage = {}
        self.keys_and_lengths = []
        self.max_size = max_pieces
        self.size = 0
        self.byte_size = 0

    def cache(self, in_matrix, out_matrix, piece_name):
        out_matrix = out_matrix.astype(bool) # convert the out_matrix to a boolean matrix
        self.output_storage[piece_name] = out_matrix
        self.input_storage[piece_name] = in_matrix
        self.keys_and_lengths.append((piece_name, len(out_matrix)))
        self.size = len(self.keys_and_lengths)
        self.byte_size += sys.getsizeof(in_matrix) + sys.getsizeof(out_matrix)

    def get(self, piece_name, start, end):
        in_matrix = self.input_storage.get(piece_name)[start:end]
        out_matrix = self.output_storage.get(piece_name)[start:end]
        return in_matrix, out_matrix.astype(np.int8)

    def save(self, save_loc="cache.pkl"):
        import pickle

        with open(save_loc, "wb") as f:
            pickle.dump(self, f)
    
    def shuffle_piece(self):
        shuffle(self.keys_and_lengths)
    