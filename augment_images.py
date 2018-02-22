import os
import csv
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import pathlib


def remove_noisy_lines(img):
	"""
		Remove black lines from the margins
	"""
	img_size = img.shape[0]
	img[0, 0:img_size] = 255 * np.ones((img_size, 3), dtype=np.uint8)
	img[-1, 0:img_size] = 255 * np.ones((img_size, 3), dtype=np.uint8)
	img[0:img_size, 0] = 255 * np.ones((img_size, 3), dtype=np.uint8)
	img[0:img_size, -1] = 255 * np.ones((img_size, 3), dtype=np.uint8)

	return img


def augment_image(image):
	"""
		Return all rotated and flipped versions of the image.

		Args:
			image (numpy.ndarray): image matrix to augment

		Returns:
			image_set (list of numpy.ndarrays): 
				list of augmented imagex
	
	"""

	image_set = []

	# rotations
	for angle in range(0, 360, 90):
		image_set.append(imutils.rotate(image, angle))

	# horizontal flips
	image_flipped_h = cv2.flip(image, 0)
	for angle in range(-90, 180, 90):
		image_set.append(imutils.rotate(image_flipped_h, angle))

	# vertical flip
	image_set.append(cv2.flip(image, 1))

	return [remove_noisy_lines(img) for img in image_set]


def process_images(read_directory = './images', write_directory = './images_extended'):
	"""
		
		Function reads initial images and augment them. Creates second
		csv file which include whole features and new names of the images.
		Creates directory to save new images if doesn't exists.

		Args:
			read_directory  (str): directory containing initial pictures
			write_directory (str): directory to save processed images
	
	"""
	
	# create output directory if doesn't exists
	try:
		pathlib.Path(write_directory).mkdir(parents=True, exist_ok=True) 
	except:
		raise Exception('Cannot create given directory!') 

	# open csv for reading metadata
	with open('decor.csv', 'rt') as reading_csv:
		data_reader = csv.reader(reading_csv, delimiter=',', quotechar='|')

		# get header - first row in a file
		header = next(data_reader)

		# open csv for creating the extended set metadata
		with open('decor_extended.csv', 'wt', newline='') as writing_csv:
			data_writer = csv.writer(writing_csv, delimiter=',', quotechar='|')

			# copy header 
			data_writer.writerow(header)

			index = 0
			
			# iterate over data samples
			for features in data_reader:

				# get last element of the list
				image_name = features[-1]

				# read the image
				image = cv2.imread(os.path.join(read_directory, image_name),
								   cv2.IMREAD_COLOR)
				
				# write the new images and update the csv
				for new_img in augment_image(image):

					new_img_name = '%.4d.png' % index
					cv2.imwrite(os.path.join(write_directory, new_img_name), new_img)

					# copy existed features of processed image 
					# and save them with a new image name 
					features[-1] = new_img_name
					data_writer.writerow(features)

					index += 1


if __name__ == "__main__":
	
	try:
		process_images()
	
	except Exception as e:
		
		print(e)
