import os
import csv
import numpy as np

reading_file_1 = 'training_labels_016.dat'
reading_file_2 = 'testing_labels_016.dat'
writing_file = 'labels_016.tsv' 

labels1_np = np.load(reading_file_1)
labels2_np = np.load(reading_file_2)
labels_np = np.concatenate([labels1_np, labels2_np], axis=0)

with open(writing_file, 'w',  newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for i in labels_np:
        writer.writerow([i])