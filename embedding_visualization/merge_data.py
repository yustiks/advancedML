import numpy as np

features_1 = np.load('training_hsv_histograms_016.dat')
features_2 = np.load('testing_hsv_histograms_016.dat')

features = np.concatenate([features_1, features_2], axis=0)
features.dump('features_embeddings_16.dat')


