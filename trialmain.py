import cv2
import numpy as np
import math
from skimage import morphology
from skimage import segmentation
from skimage.measure import label
from drawcircles import draw_circles
from averageCircleRadius import get_average

########################## rbc extraction  #######################################################################
original_img = cv2.imread('Images/malaria1.jpg', cv2.IMREAD_COLOR)
original_img = cv2.resize(original_img, (300, 300))
cv2.imshow('original image', original_img)
width, height, depth = original_img.shape
gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
median_filtered_image = cv2.medianBlur(gray_image, 5)
cv2.imshow('median filter', median_filtered_image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
adaptive_histogram = clahe.apply(gray_image)
cv2.imshow('adaptive histogram', adaptive_histogram)
chan_vese_segmentation = segmentation.chan_vese(adaptive_histogram)
chan_vese_segmentation_output = np.zeros((width,height),np.uint8)
chan_vese_segmentation_output[chan_vese_segmentation == True] = 255
cv2.imshow('chan vese', chan_vese_segmentation_output)
labels = label(chan_vese_segmentation_output)
largestCC = labels == np.argmax(np.bincount(labels.flat))
largest_connected_component = np.zeros((width,height),np.uint8)
largest_connected_component[largestCC==True] = 255
holes_filled = largest_connected_component
cv2.imshow('holes filled', holes_filled)
circles = cv2.HoughCircles(holes_filled,cv2.HOUGH_GRADIENT,1,15,
                           param1=100,param2=9,minRadius=12,maxRadius=22)
circles = np.uint16(np.around(circles))
cimg = draw_circles(holes_filled, circles)
cv2.imshow('circles',cimg)
black_area=0
for i in range(width):
    for j in range(height):
        if cimg[i][j][0]==0 and cimg[i][j][1]==0 and  cimg[i][j][2] == 0:
            black_area+=1

total_no_of_rbc = int(black_area//get_average(circles))
print(total_no_of_rbc)

####################### Parasite Extraction###################################################################


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


