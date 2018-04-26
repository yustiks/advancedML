import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


TRAINING_SIZE = 386
TESTING_SIZE = 99
DATA_SIZE = TRAINING_SIZE + TESTING_SIZE

IMG_SIZE = 150
CHANNELS = 3

PICS_PER_ROW = int(math.sqrt(DATA_SIZE - 0.01)) + 1
SPRITE_SIZE = PICS_PER_ROW * IMG_SIZE

sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE, CHANNELS), dtype=np.uint8)

for i in range(TRAINING_SIZE):
	
	print("Spriting " + str(i))

	# read the image
	img_name = "%.4d.png" % (8 * i)
	img_path = os.path.join('..', 'by_style', 'training', img_name)
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)

	# assign the image to the sprite
	row = i // PICS_PER_ROW
	col = i %  PICS_PER_ROW
	sprite[row * IMG_SIZE:(row + 1) * IMG_SIZE,
	       col * IMG_SIZE:(col + 1) * IMG_SIZE, :] = img


for i in range(TESTING_SIZE):
	
	index = TRAINING_SIZE + i
	print("Spriting " + str(index))

	# read the image
	img_name = "%.4d.png" % (8 * i)
	img_path = os.path.join('..', 'by_style', 'testing', img_name)
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)

	# assign the image to the sprite
	row = index // PICS_PER_ROW
	col = index %  PICS_PER_ROW
	sprite[row * IMG_SIZE:(row + 1) * IMG_SIZE,
	       col * IMG_SIZE:(col + 1) * IMG_SIZE, :] = img


# save image to disk
cv2.imwrite('sprite.png', sprite)