import cv2
import skimage
import numpy as np
import math
from skimage import morphology
from skimage import segmentation
from non_max_suppression import orientated_non_max_suppression
print(skimage.__version__)
original_img = cv2.imread('Images/malaria2.jpg', cv2.IMREAD_COLOR)
original_img = cv2.resize(original_img, (300, 300))
cv2.imshow('original image', original_img)


################################################### RBC EXTRACTION ############################################

width, height, depth = original_img.shape
gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
histogram_equalized = cv2.equalizeHist(gray_image)
cv2.imshow('histogram equalized', histogram_equalized)
chan_vese_segmentation = segmentation.chan_vese(histogram_equalized)
###############################################################################################################

################ parasite extracion#########################################################3

blue_plane = np.zeros((width,height),np.uint8)
for i in range(width):
    for j in range(height):
        extracted_pixel = original_img[:,:,0][i][j] - math.ceil((original_img[:, :, 1])[i][j]/2)-math.ceil(original_img[:, :, 2][i][j]/2)
        if extracted_pixel > 0:
            blue_plane[i][j] = extracted_pixel
cv2.imshow('parasite extraction', blue_plane)
blue_plane[blue_plane > 29] = 255
blue_plane[blue_plane < 29] = 0
cv2.imshow('parasite extraction 2', blue_plane)
extracted_Integer_Image = np.zeros((width, height), np.int32)
extracted_Integer_Image = blue_plane
im_floodfill = blue_plane.copy()
mask2 = np.zeros((width + 2, height + 2), np.uint8)
cv2.floodFill(im_floodfill, mask2, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
fill_parasite_holes = blue_plane | im_floodfill_inv
cv2.imshow('filled holes', fill_parasite_holes)
labeled_image = morphology.label(extracted_Integer_Image)
noiseless_label = morphology.remove_small_objects(labeled_image, min_size=100, in_place=False)
noise_removal = np.zeros((width,height), np.uint8)
noise_removal[noiseless_label > 0] = 255
cv2.imshow('noise removal', noise_removal)
labels, label_array = cv2.connectedComponents(noise_removal)
if np.max(label_array)==0:
    noise_removal = blue_plane
    labels, label_array = cv2.connectedComponents(blue_plane)



cv2.waitKey(0)
cv2.destroyAllWindows()


