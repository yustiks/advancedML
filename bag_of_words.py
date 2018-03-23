import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
	
	datasets_path = ('./by_style/', './by_country/', './by_type/' )

	for path in datasets_path:

		testing = pd.read_csv(os.path.join(path, 'testing.csv'), sep=',', header=0)

		training = pd.read_csv(os.path.join(path, 'testing.csv'), sep=',', header=0)

		for index, row in training.iterrows(): 
			
			img_path = os.path.join(path, 'testing', row[0])

			print(img_path)
			img = cv2.imread(img_path)
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			sift = cv2.xfeatures2d.SIFT_create()

			kp = sift.detect(gray, None)

			img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

			plt.imshow(img)
			kp, des = sift.detectAndCompute(gray,None)

			print(len(kp), len(des))

			cv2.imwrite('/home/dawid/sift_keypoints.jpg',des[3])

			#sys.exit(2)

