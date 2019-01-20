import model_tb as model
import data
import argparse
import os
import pickle
import tensorflow as tf

if __name__ == '__main__':
    
    # Check for the existence of previous cache and models
    parser = argparse.ArgumentParser(description="RNN predict")
    parser.add_argument(
        "--model_name", help="name of pre-trained model.", required=True)
    parser.add_argument(
        "--cache_dir", help="Path to cache file",
        default='../output/cache.pkl')
    
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
    
    # Building model
    print('Building model')
    music_model = model.biaxial_model(t_layer_sizes=[300,300], n_layer_sizes=[100,50], trainer=tf.train.AdamOptimizer())
    
    print('Start predicting')
    music_model.predict(cache,model_name,step=320,conservativity=1,n=50,saveto='../output/predict songs')
