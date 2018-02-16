import os
import csv
import cv2
import imutils
import matplotlib.pyplot as plt

read_directory = './images'
write_directory = './images_extended'


def augment_image(image):
	"""Return all rotated and flipped versions of the image"""

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
	
	return image_set


# open csv for reading metadata
with open('decor.csv', 'rt') as reading_csv:
	data_reader = csv.reader(reading_csv, delimiter=' ', quotechar='|')

	# open csv for creating the extended set metadata
	with open('decor_extended.csv', 'wt', newline='') as writing_csv:
		data_writer = csv.writer(writing_csv, delimiter=' ', quotechar='|')

		index = 0
		# iterate over data samples
		for row in data_reader:
			print(row)

			# extract the features
			features = row[0].split(',')
			image_name = features[-1]

			# skip the header row
			if image_name == 'file':
				data_writer.writerow(row)
				continue

			# read the image
			image = cv2.imread(os.path.join(read_directory, image_name),
				               cv2.IMREAD_COLOR)

			# write the new images and update the csv
			for img in augment_image(image):

				image_index = '%.4d.png' % index
				cv2.imwrite(os.path.join(write_directory, image_index), img)

				features[-1] = image_index
				new_features = [','.join(features)]
				
				data_writer.writerow(new_features)
				index += 1
