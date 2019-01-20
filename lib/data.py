# For data preprocessing, we just used the codes provided by Daniel Johnson here: https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/data.py

import itertools
import os
import random
import sys

from cache import Cache
from midi_to_statematrix import *


def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0] * 66 + [1]

    return [noteSentinel(note) for note in range(upperBound - lowerBound)]


def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def buildContext(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lowerBound) % 12
            context[pitchclass] += 1
    return context


def buildBeat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    part_prev_vicinity = list(
        itertools.chain.from_iterable((getOrDefault(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]


def noteStateSingleToInputForm(state, time):
    beat = buildBeat(time)
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat) for note in range(len(state))]


def noteStateMatrixToInputForm(statematrix):
    inputform = np.int8([noteStateSingleToInputForm(state, time) for time, state in enumerate(statematrix)])
    return inputform

def getpices(path='midis', midi_len=128, mode='all',composer=None):
    pieces = {}
    if not os.path.exists(path):
        # Download midi files
        import midi_scraper
    song_count = 0

    for composer_name in os.listdir(path):
        if composer is not None and composer_name not in composer: continue
        for fname in os.listdir(path+'/'+composer_name):
            if fname[-4:] not in ('.mid','.MID'):
                continue

            name = fname[:-4]

            try:
                outMatrix = midiToNoteStateMatrix(os.path.join(path, composer_name, fname))
            except:
                continue
            if len(outMatrix) < midi_len:
                continue

            pieces[name] = np.int8(outMatrix)
            song_count += 1
            print ("Loaded {}-{}".format(composer_name, fname))
            if mode != 'all':
                if song_count >= 10:
                    print ("{} songs are loaded".format(song_count))
                    return pieces
    print ("{} songs are loaded".format(song_count))
    return pieces

def getPieceSegmentFaulty(pieces, piece_length=128, measure_len=16, validation=False):
    # piece_length means the number of ticks in a training sample, measure_len means number of ticks in a measure
    val_size = len(pieces) // 5
    if validation:
        pieces_set = pieces[-val_size:]
    else:
        pieces_set = pieces[:-val_size]
    piece_name, full_length = random.choice(pieces)

    # We just need a segment of a piece as train data, and we want the start of a sample is the start of a measure
    start = random.randrange(0, full_length-piece_length,measure_len)

    seg_in, seg_out = cache.get(piece_name, start, start+piece_length)

    return seg_in, seg_out

def generate_batch(cache, batch_size, piece_length=128):
    while True:
        i,o = zip(*[getPieceSegment(cache, piece_length) for _ in range(batch_size)])
        yield(i,o)

def generate_val_batch(cache, batch_size, piece_length=128):
    while True:
        i,o = zip(*[getPieceSegment(cache, piece_length,validation=True) for _ in range(batch_size)])
        yield(i,o)

def getPieceSegment(cache, piece_length=128, measure_len=16, validation=False):
    # piece_length means the number of ticks in a training sample, measure_len means number of ticks in a measure
    val_size = max(cache.size // 10, 1)
    if validation:
        keys_and_lengths = cache.keys_and_lengths[-val_size:]
    else:
        keys_and_lengths = cache.keys_and_lengths[:-val_size]
    piece_name, full_length = random.choice(keys_and_lengths)

    # We just need a segment of a piece as train data, and we want the start of a sample is the start of a measure
    start = random.randrange(0, full_length-piece_length,measure_len)

    seg_in, seg_out = cache.get(piece_name, start, start+piece_length)

    return seg_in, seg_out


def initialize_cache(pieces, piece_length=128, measure_len=16, save_loc="cache.pkl"):
    midi_cache = Cache()
    for piece_name in pieces:
        out_matrix = pieces[piece_name]
        print(out_matrix.shape)
        in_matrix = noteStateMatrixToInputForm(out_matrix)
        midi_cache.cache(in_matrix, out_matrix, piece_name)

    print("Cache initialized with {} pieces; total size is {} bytes".format(len(pieces), midi_cache.byte_size))
    midi_cache.shuffle_piece()
    midi_cache.save(save_loc=save_loc)
    return midi_cache


def translate(note_matrix, direction="up"):
    """
    Translate the notes in a piece up or down one note to test invariance.
    Preprocess the note_matrix to get mute the highest and lowest note.
    """
    translated_matrix = []

    # If we translate upwards, the highest note falls off
    # If we translate down, the lowest note falls off
    dropped_idx = 0
    if direction == "up":
        dropped_idx = -1

    for step_notes in note_matrix:
        # Each step_notes is a (78 x 2)
        if direction == "up":
            translated_matrix.append([[0, 0]] + [pair for pair in step_notes[0:-1]])
        else:
            translated_matrix.append([pair for pair in step_notes[1:]] + [[0, 0]])

    return translated_matrix

def translate_np(note_matrix, shift=1, direction="up"):
    """
    Translate the notes in a piece up or down one note to test invariance.
    Preprocess the note_matrix to get mute the highest and lowest note.
    """
    if direction == "down":
        shift = shift * -1
    translated_matrix = np.roll(note_matrix, shift, axis=-2)
    return translated_matrix
