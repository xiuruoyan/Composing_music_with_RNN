import itertools

import numpy as np
import tensorflow as tf

lowerbound = 24
upperbound = 102

# Python Operations
def start_sentinel():
    def note_sentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lowerbound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0]*66 + [1]
    return [note_sentinel(note) for note in range(upperBound-lowerbound)]


def get_or_default(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def build_beat(time):
    return [2*((time // 2**x) % 2)-1 for x in range(4)]


def build_context(output_chord):
    # state is the output state for the network
    # [batch_size, ]
    context = [0 for _ in range(12)]
    for note, note_state in enumerate(output_chord):
        if note_state[0] == 1:
            pitchclass = (note + lowerbound) % 12
            context[pitchclass] += 1
    return context


def build_note_input(note, state, context, beat):
    part_position = [note]

    pitchclass = (note + lowerbound) % 12

    pitchclass_one_hot = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable((get_or_default(state, note+i, [0,0]) for i in range(-12, 13))))

    # Rearrange context so that pitch classes are
    context_part = context[pitchclass:] + context[:pitchclass]

    return part_position + pitchclass_one_hot + part_prev_vicinity + context_part + beat + [0]


def map_output_to_input(state, time):
    beat = build_beat(time)

    # Need to convert output state from np.array to list of tuples
    state = state.tolist()
    context = build_context(state)

    # Needs to return a numpy array in order to be wrapped as a tensorflow op
    return np.array([build_note_input(note, state, context, beat) for note in range(len(state))], dtype=np.float32)


# Tensorflow Graph operations
def map_output_to_input_tf(out_notes, time):
    """
    Tensorflow operation for transforming output notes to input vectors
    map_output_to_input will be passed args as numpy arrays and numpy scalars
    Args:
        out_notes: tensor of shape (notes x 2)
        time: scalar tensor for the time
    """
    return tf.py_func(map_output_to_input, [out_notes, time], tf.float32)
