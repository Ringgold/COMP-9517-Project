import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

def to_binary_image(input_image: np.array, background_potion = 6):
    input = input_image.copy()
    meaningful_range = np.max(input) - np.min(input)
    input = cv2.normalize(input, None, alpha=0, beta=meaningful_range, norm_type=cv2.NORM_MINMAX)
    input[input >= meaningful_range/background_potion] = 255
    input[input < meaningful_range/background_potion] = 0
    return input

def dilate_erode_reduce_noise(input_binary: np.array, kernel = (3,3), iteration = 5):
    '''
    Try Dilate and erode
    1. Dilate the objects first to get rid of the noise inside each objects
    2. erode the objects with same degree to decrease the size to original state
    '''
    input = input_binary.copy()
    input = cv2.dilate(input, kernel, iterations = iteration)
    input = cv2.erode(input, kernel, iterations = iteration)
    
    return input

def find_coutours_draw(input_gray: np.array):
    '''
    Find all of the coutours of the input image cell and draw the coutours
    '''
    threshold = (np.max(input_gray) - np.min(input_gray))/2
    input_edges = cv2.Canny(input_gray, threshold, threshold)
    plt.title("input_edges")
    plt.imshow(input_edges, cmap="gray")
    plt.show()

    # pixel_norm_range = np.max(input_gray) - np.min(input_gray)
    # input_norm = cv2.normalize(cell, None, alpha=0, beta=pixel_norm_range, norm_type=cv2.NORM_MINMAX)

    # ret, thresh = cv2.threshold(input_gray, math.floor(pixel_norm_range/3), np.max(input_norm), np.min(input_norm))
    # plt.title("input_gray")
    # plt.imshow(input_gray, cmap="gray", vmin=0, vmax=pixel_norm_range)
    # plt.show()

    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("Number of contours = {}".format(str(len(contours))))
    # print('contours {}'.format(contours[0]))

    # # cv2.drawContours(input_gray, contours, -1, (0, 255, 0), 2)
    # cv2.drawContours(input_gray, contours, -1, (0, 255, 0), 3)

    # # cv2.imshow('Image', img)
    # cv2.imshow('Image GRAY', input_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # ret, thresh = cv2.threshold(input_gray, cv2.THRESH_BINARY, 255, 0)
    # print(ret, thresh)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # thresh1 = cv2.adaptiveThreshold(input_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                       cv2.THRESH_BINARY, 199, 5)
  
    # thresh2 = cv2.adaptiveThreshold(input_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                       cv2.THRESH_BINARY, 199, 5)

    

    # # apply binary thresholding
    # ret, thresh = cv2.threshold(input_gray, 150, 255, cv2.THRESH_BINARY)
    # # visualize the binary image
    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite('image_thres1.jpg', thresh)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get image root path
    image_src_path = str(Path(__file__).parent.resolve()) + str("\\Sequences")
    # individual image path example
    path_cell = str(image_src_path + "\\01\\t000.tif")
    path_seg = str(image_src_path + "\\01_GT\\SEG\\man_seg088.tif")
    path_tra = str(image_src_path + "\\01_GT\\TRA\\man_track013.tif")
    # Read as gray
    cell = cv2.imread(path_cell, cv2.IMREAD_GRAYSCALE)
    cell_seg = cv2.imread(path_seg, -1)
    cell_tra = cv2.imread(path_tra, -1)

    # Turn into np array first
    cell = np.array(cell)
    plt.title("cell")
    plt.imshow(cell, cmap="gray")
    plt.show()
    # To binary
    cell_binary = to_binary_image(cell)
    # reduce noise
    cell_binary_no_noise = dilate_erode_reduce_noise(cell_binary)
    plt.title("cell_binary_no_noise")
    plt.imshow(cell_binary_no_noise, cmap="gray")
    plt.show()
    # get coutours
    cell_binary_coutour = find_coutours_draw(cell_binary_no_noise)
    plt.title("cell_binary_coutour")
    plt.imshow(cell_binary_coutour, cmap="gray")
    plt.show()
    
    '''
    Get meaningful range of the original image:
    From what I see, the original cells' images are just meaningful from 128 to 136 which is 8 numbers
    '''
    # meaningful_range = np.max(cell) - np.min(cell)
    # cell_norm = cv2.normalize(cell, None, alpha=0, beta=meaningful_range, norm_type=cv2.NORM_MINMAX)
    # print(cell_norm.shape, meaningful_range)
    # plt.title("cell_norm")
    # plt.imshow(cell_norm, cmap="gray", vmin=0, vmax=meaningful_range)
    # plt.show()

    # Check SEG
    # print(cell_seg.shape)
    # plt.title("cell_seg")
    # plt.imshow(cell_seg)
    # plt.show()

    # Check TRA
    # plt.title("cell_tra")
    # plt.imshow(cell_tra)
    # plt.show()
    