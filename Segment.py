import numpy as np
import cv2
import random

import sys
from matplotlib import pyplot as plt
from skimage.segmentation import  clear_border
from skimage import  color
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def convertImage(image, target_type_min, target_type_max):
    new_img = np.zeros_like(image)
    cv2.normalize(src = image,dst=new_img,alpha=target_type_min,beta=target_type_max,norm_type=cv2.NORM_MINMAX)
    new_img = new_img.astype(np.uint8)

    return new_img


# def applyWatershed(image):
#     thresh = cv2.GaussianBlur(image,(1,1),0)
#     ret,thresh = cv2.threshold(thresh,0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
#
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#
#     # clear cells on border
#     opening = clear_border(opening)
#
#     # sure background
#     sure_bg = cv2.dilate(opening, kernel, iterations=10)
#
#     # sure foreground
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
#
#     # finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
#
#     #marker
#     _, markers = cv2.connectedComponents(sure_fg)
#
#     markers = markers + 1
#     markers[unknown == 255] = 0
#
#     img  = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     markers = cv2.watershed(img, markers)
#
#     mask = np.zeros_like(image)
#     color_img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
#
#     # img2 = color.label2rgb(markers,bg_label =1)
#     mask[markers== -1] = 255
#     mask[markers == 0] = 255
#
#     return mask,color_img
#
def doWatershed(image):
    blur = cv2.GaussianBlur(image, (3,3), 0)

    # thresholding
    ret, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    plt.imshow(th1)
    plt.show()
    # dist transform
    D = ndimage.distance_transform_edt(th1)

    # markers
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=th1)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    # apply watershed
    labels = watershed(-D, markers, mask=th1)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    return labels

img_path = "./Sequences/01/t000.tif"
print(img_path)
image  = cv2.imread(img_path,-1)
image = convertImage(image,0,255)
#
# mask,img = applyWatershed(image)
# plt.imshow(mask)
# plt.show()
# _,contours,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for contour in range(len(contours)):
#     cv2.drawContours(img,contours,contour,np.random.randint(0,256,3).tolist(),5)
#
#
# plt.imshow(img,cmap='gray')
# plt.show()



labels = doWatershed(image)
color_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
for label in np.unique(labels):
    if label == 0:
        continue

    # draw label on the mask
    mask = np.zeros_like(image)
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    _,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(color_image, cnts, -1,np.random.randint(0,256,3).tolist(), 5)
plt.imshow(color_image)
plt.show()
