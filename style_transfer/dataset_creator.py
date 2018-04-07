import os
import sys
import cv2
import csv
import pathlib
import numpy as np


def create_data():
	
	with open(os.path.join('..', 'decor_extended.csv'), 'rt', newline='') \
		as reading_csv:
		data_reader = csv.reader(reading_csv, delimiter=',', quotechar='|')

		# ignore the header
		header = next(data_reader)

		# dictionary to hold image names for each style
		style_images = {}

		for data_line in data_reader:

			# unpack values
			style = data_line[3].replace(' ', '_')
			img_name = data_line[6]

			# append image to its corresponding style
			style_images.setdefault(style, []).append(img_name)

		print(style_images)


if __name__ == "__main__":

	print('Creating styles dataset...')
	create_data()
	print('Styles dataset is ready!')
