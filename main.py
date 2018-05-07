import cv2
import numpy as np
from skimage import morphology
from non_max_suppression import orientated_non_max_suppression

original_img = cv2.imread('Images/malaria1.jpg', cv2.IMREAD_COLOR)
original_img = cv2.resize(original_img, (300, 300))
gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

median_filtered_image = cv2.medianBlur(gray_image, 5)
histogram_equalized = cv2.equalizeHist(median_filtered_image)
extracted_image = np.zeros((300, 300), np.uint8)
cv2.imshow('extracted_image0', extracted_image)
width, height, depth = original_img.shape

for i in range(width):
    for j in range(height):
        if (original_img[i, j, 2] <= 200 and original_img[i, j, 2] >= 75) and (
                original_img[i, j, 1] <= 150 and original_img[i, j, 1] >= 35) and (
                original_img[i, j, 0] <= 235 and original_img[i, j, 0] >= 175):
            extracted_image[i, j] = median_filtered_image[i, j]

blur = cv2.GaussianBlur(extracted_image, (5, 5), 0)

ret3, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

noiseless_binary = np.zeros((300, 300), np.uint8)

cv2.fastNlMeansDenoising(binary_image, noiseless_binary, 45, 7, 21)

intensity_adjusted = np.ones((300, 300), np.uint8)  # print np.max(histogram_equalized)# print type(histogram_equalized)
intensity_adjusted = histogram_equalized / 125.0

ret4, binary_image2 = cv2.threshold(intensity_adjusted, 0.78, 1, cv2.THRESH_BINARY)

area_opened_image = cv2.morphologyEx(binary_image2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

sobelx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)  # threshold
fudgefactor = 0.5
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0  # non - maximal suppression
mag = orientated_non_max_suppression(mag, ang)
# # alternative but doesn 't consider gradient direction#
# mag = skimage.morphology.thin(mag.astype(np.bool)).astype(np.float32)

# create mask
mag[mag > 0] = 255
edge_image = mag.astype(np.uint8)
dilated_image = np.zeros((300, 300), np.uint8)

for i in range(1, width - 1):
    for j in range(1, height - 1):
        dilated_image[i, j] = max(edge_image[i, j + 1], max(edge_image[i, j], edge_image[i, j - 1]))
        dilated_image[i, j] = max(edge_image[i + 1, j], max(edge_image[i - 1, j], dilated_image[i, j]))

im_floodfill = dilated_image.copy()

mask = np.zeros((height + 2, width + 2), np.uint8)

cv2.floodFill(im_floodfill, mask, (0, 0), 255);

im_floodfill_inv = cv2.bitwise_not(im_floodfill)

im_out = dilated_image | im_floodfill_inv

cv2.imshow('im_floodfill', im_floodfill)
cv2.imshow('im_floodfill_inv', im_floodfill_inv)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
adaptive_histogram = clahe.apply(gray_image)
ret4, binary_image3 = cv2.threshold(adaptive_histogram, 0, 255, cv2.THRESH_OTSU)

im_floodfill2 = binary_image3.copy()

mask2 = np.zeros((height + 2, width + 2), np.uint8)

cv2.floodFill(im_floodfill2, mask2, (0, 0), 255)
cv2.imshow('im_floodfill2', im_floodfill2)
im_floodfill_inv2 = cv2.bitwise_not(im_floodfill2)
cv2.imshow('im_floodfill_inv2', im_floodfill_inv2)
im_out2 = binary_image3 | im_floodfill_inv

opening = cv2.morphologyEx(im_out2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

binary_opened = morphology.remove_small_objects(opening, min_size=40, connectivity=2)

# cv2.imshow('binary_image3', binary_image3)
# cv2.imshow('im_out2', im_out2)
# cv2.imshow('opening', opening)
# cv2.imshow('binary_opened', binary_opened)
# cv2.imshow('image', original_img)
# cv2.imshow('gray_image', gray_image)
# cv2.imshow('median_filtered_image', median_filtered_image)
# cv2.imshow('histogram_equalized', histogram_equalized)
# cv2.imshow('extracted_image', extracted_image)
# cv2.imshow('binary_image', binary_image)
# cv2.imshow('noiseless_binary', noiseless_binary)
# cv2.imshow('intensity_adjusted', intensity_adjusted)
# cv2.imshow('binary_image2', binary_image2)
# cv2.imshow('area_opened_image', area_opened_image)
# cv2.imshow('edge_image', edge_image)
# cv2.imshow('dilated_image', dilated_image)
# cv2.imshow("Floodfilled Image", im_floodfill)
# cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# cv2.imshow("Foreground", im_out)
# cv2.imshow('adaptive histogram', adaptive_histogram)

cv2.waitKey(0)
cv2.destroyAllWindows()
