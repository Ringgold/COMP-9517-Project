import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from pathlib import Path
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def max_filter(img, N):
    x = (N - 1) / 2
    A = np.zeros(shape = np.shape(img))
    row, col = np.shape(img)
    for i in range(0,row):
        for j in range(0,col):
            row_min = int(max(i-x,0))
            row_max = int(min(i+x,row))
            col_min = int(max(j-x,0))
            col_max = int(min(j+x,col))
            A[i][j] = np.max(img[row_min:row_max,col_min:col_max])
    return A

def min_filter(img, N):
    kernel = np.ones((N, N), np.uint8)
    B = cv2.erode(img, kernel, iterations=1)
    return B

def myNormalize(img):
    streched = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return streched

def apply_watershed(img, threshold=None):
    _, img_array = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    distance = myNormalize(cv2.distanceTransform(img_array, distanceType = 2, maskSize = 0))
    '''plt.title("distance")
    plt.imshow(distance, cmap = 'gray', vmin = 0, vmax = 255)
    plt.show()'''
    local_max = peak_local_max(distance.copy(), footprint=np.ones((3,3)), labels = img_array)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_max.T)] = True
    mask = np.uint8(mask)
    markers, _ = ndi.label(mask)
    ws_labels = watershed(-distance, markers, mask = img_array)
    return ws_labels, distance

def process(path_cell, N):
    cell = cv2.imread(path_cell, cv2.IMREAD_GRAYSCALE)
    
    cell_normalized = myNormalize(cell)
    '''plt.title("original")
    plt.imshow(cell_normalized, cmap = 'gray', vmin = 0, vmax = 255)
    plt.show()'''
    cell_min_filtered = min_filter(cell_normalized, 3)
    cell_watersheded, _ = apply_watershed(cell_min_filtered, np.mean(cell_normalized))

    cell_finalized = myNormalize(cell_watersheded)
    cv2.imwrite(f"./streched/{N}.png", cell_finalized)
    '''plt.title("ws_labels")
    plt.imshow(cell_finalized)
    plt.show()'''
    


def format_name_digits(number: int):
    number_str = '000'
    if number < 10:
        number_str = '00' + str(number)
    elif number >= 10 and number <= 99: 
        number_str = '0' + str(number)
    elif number >= 100 and number <= 999: 
        number_str = str(number)
    else:
        print("TOO BIG NUMBER for format_name_digits:", number)
    return number_str


if __name__ == "__main__":
    # Get image root path
    image_src_path = str(Path(__file__).parent.resolve()) + str("\\Sequences")
    # individual image path example\
    path_divide = "\\"
    path_cell_subfolder = "01"
    path_cell_name = "t000.tif"
    path_cell = str(image_src_path + path_divide + path_cell_subfolder + path_divide + path_cell_name)
    # Do single image segmenting
    # get_cell_segment_info(path_cell)

    # Do batch images segmenting
    for i in range(0, 60):
        path_cell_subfolder = "01"
        path_cell_name = "t" + format_name_digits(i) + ".tif"
        path_cell = str(image_src_path + path_divide + path_cell_subfolder + path_divide + path_cell_name)
        process(path_cell, i)