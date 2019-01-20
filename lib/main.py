import model_tb as model
import data

import os
import pickle
import sys
import tensorflow as tf
import argparse

if __name__ == '__main__':

    # Check for the existence of previous cache and models
    parser = argparse.ArgumentParser(description="RNN train")
    parser.add_argument(
        "--cache_dir", help="Path to cache file",
        default='../output/cache.pkl')
    parser.add_argument(
        "--model_name", help="Path to pretrained model.", default=None)
    args = parser.parse_args()
    
    cache_name = args.cache_dir
    model_name = args.model_name
    

    if not os.path.exists(cache_name):
        composers = input("Enter composers separated by spaces, no input means all music: ").split()
        all_pieces = {}
        
        if len(composers)==0:
            all_pieces.update(data.getpices(path="../data/midis", mode='all'))
        elif composers[0] == 'pop':
            all_pieces.update(data.getpices(path='../data/pop_midis', mode='all'))
        else:
            for c in composers:
                all_pieces.update(data.getpices(path="../data/midis", composer=c))

        cache = data.initialize_cache(all_pieces, save_loc=cache_name)
    else:
        with open(cache_name, 'rb') as f:
            cache = pickle.load(f)

    # Build and load the pre-existing model if it exists
    print('Building model')
    music_model = model.biaxial_model(t_layer_sizes=[300,300],
        n_layer_sizes=[100,50],
        trainer = tf.train.AdamOptimizer())

    print('Start training')
    music_model.train(cache, batch_size=5, max_epoch=50000,
        predict_freq=100, pre_trained_model=model_name)
