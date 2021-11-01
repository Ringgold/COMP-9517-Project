import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.segmentation
from skimage import color
from pathlib import Path

# class Cell
# color cell

markers_color_value_offset = 10

def get_binary(input_image: np.array, min_cell_pixel_portion = 1/8, threshold_block_size = 37):
    input = input_image.copy()
    # Get meaningful range
    meaningful_range = np.max(input) - np.min(input)
    # Assume certain small portion of the max_pixel_value is the minimum cell pixel value
    min_cell_pixel = math.ceil(meaningful_range * min_cell_pixel_portion)
    # normalize image based on the acquired meaningful range
    input_norm = cv2.normalize(input, None, alpha=0, beta=meaningful_range, norm_type=cv2.NORM_MINMAX)
    input_norm[input_norm < min_cell_pixel] = 0
    input_binary = cv2.adaptiveThreshold(
        input_norm,meaningful_range,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        threshold_block_size,
        0
    )
    return input_binary

def get_reduce_noise_by_closing(input_binary: np.array, kernel_size_closing = (5,5), iterations = 5):
    '''
    Try dilate and erode: closing
    1. Dilate the objects first to get rid of the noise inside each objects
    2. Erode the objects with same degree to decrease the size to original state
    '''
    input_closing = input_binary.copy()
    kernel = np.ones(kernel_size_closing,np.uint8)
    input_closing = cv2.morphologyEx(input_closing, cv2.MORPH_CLOSE, kernel, iterations)
    return input_closing

def get_reduce_noise_by_opening(input_binary: np.array, kernel_size_opening = (7,7), iterations = 5):
    '''
    Try erode and dilate: opening
    Erode then dilate
    '''
    input_opening = input_binary.copy()
    kernel = np.ones(kernel_size_opening,np.uint8)
    input_opening = cv2.morphologyEx(input_opening, cv2.MORPH_OPEN, kernel, iterations)
    return input_opening

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def find_contours(
        input_image: np.array,
        input_binary: np.array, 
        kernel_size_bg = (3, 3),
        iterations_bg = 7
    ):
    global markers_color_value_offset
    '''
    Find all of the contours of the input image cell and draw the contours
    '''
    kernel = np.ones(kernel_size_bg, np.uint8)
    sure_bg = cv2.dilate(input_binary, kernel, iterations=iterations_bg)
    dist_transform = cv2.distanceTransform(input_binary,cv2.DIST_MASK_3,5)
    # plt.title("dist_transform")
    # plt.imshow(dist_transform)
    # plt.show()
    coords = peak_local_max(
        dist_transform, 
        footprint=np.ones((36, 36), np.uint8)
    )
    sure_fg = np.zeros(dist_transform.shape, dtype=bool)
    sure_fg[tuple(coords.T)] = True
    sure_fg = np.uint8(sure_fg)
    markers, number_feature = ndi.label(sure_fg)
    plt.title("markers")
    plt.imshow(markers)
    plt.show()
    ws_labels = skimage.morphology.watershed(-dist_transform, markers, mask=input_binary)
    plt.title("ws_labels")
    plt.imshow(ws_labels)
    plt.show()

    # unknown = cv2.subtract(sure_bg, sure_fg)
    # plt.title("sure_fg")
    # plt.imshow(sure_fg)
    # plt.show()

    # markers_amount, markers = cv2.connectedComponents(sure_fg)
    
    # plt.title("markers")
    # plt.imshow(markers)
    # plt.show()
    # '''
    # object_amount_not_on_borders:
    # 1. The total amount of objects detected in the image without considering the ones on borders
    # 2. Need to use markers_amount - 1 since markers_amount includes the background
    # '''
    # object_amount_not_on_borders = markers_amount - 1

    # # make sure the background are not set as 0 and consider as unsured area
    # markers = markers + markers_color_value_offset
    # max_unkown = np.max(unknown)
    # markers[unknown==max_unkown] = 0
    # watershed = cv2.watershed(input_image, markers)

    # # watershed boundaries are -1
    # edges = np.zeros(input_binary.shape, np.uint8)
    # edges[watershed == -1] = 1

    # #label2rgb - Return an RGB image where color-coded labels are painted over the image.
    # colored_segmentation = color.label2rgb(watershed, bg_label=0)

    # return watershed, edges, colored_segmentation, object_amount_not_on_borders

def get_object_pixel_record(object_color_array):
    # Get useful pixel values
    unique_objects = np.unique(object_color_array)
    unique_objects = unique_objects[unique_objects >= 1] 

    # initialize the pixel dict with all object count starting from 0
    object_dict = dict.fromkeys(unique_objects)
    for key in object_dict:
        object_dict[key] = 0

    # get the pixel value count for each object key label
    for row in object_color_array:
        for col in row:
            if col >= 1:
                object_dict[col] += 1
    return object_dict

def get_cell_segment_info(path_cell: str):
    # Read as gray
    cell_original = cv2.imread(path_cell)
    cell = cv2.imread(path_cell, cv2.IMREAD_GRAYSCALE)

    # Turn into np array first
    cell = np.array(cell)

    # Grey image to binary
    cell_binary = get_binary(cell)

    # reduce noise
    cell_binary_opening = get_reduce_noise_by_opening(cell_binary)
    cell_no_noise = cell_binary_opening.copy()

    # removing border interfered cells
    # cell_no_border = skimage.segmentation.clear_border(cell_no_noise)

    # get contours
    find_contours(cell_original, cell_no_noise)
    # cell_watershed, cell_edges, cell_colored_segmentation, cell_amount_not_on_borders = find_contours(cell_original, cell_no_noise)
    # # After minus markers_color_value_offset, all of the objects color pixel will have value >= 1
    # cell_watershed = cell_watershed - markers_color_value_offset
    # cell_pixel_dict = get_object_pixel_record(cell_watershed)
    # cell_pixel_array = [v for _, v in cell_pixel_dict.items() if v >= 0]
    # cell_pixel_size_average = round(sum(cell_pixel_array) / len(cell_pixel_array))

    # # Print cell detection result for a single image
    # all_figures = plt.figure(figsize = (13,13))
    # titles = ['cell_original','cell_no_noise','cell_edges','cell_colored_segmentation']
    # images = [cell, cell_no_noise, cell_edges, cell_colored_segmentation]
    # for i in range(4):
    #     ax1 = all_figures.add_subplot(2,2,i+1)
    #     if i == 3:
    #         ax1.imshow(images[i])
    #     else:
    #         ax1.imshow(images[i], cmap="gray")
    #     ax1.set_title(titles[i])
    #     ax1.set_axis_off()
    # suptitle = "Cells count: " + str(cell_amount_not_on_borders) + ", average pixel size: " + str(cell_pixel_size_average) +\
    #     ", img:" + path_cell_subfolder + '.' + path_cell_name
    # plt.suptitle(suptitle)
    # # plt.tight_layout()
    # # output_subfolder_name = "output1-1/"
    # # plt.savefig(
    # #     output_subfolder_name +\
    # #         path_cell_subfolder + '_' + path_cell_name + \
    # #         ", Cells count_" + str(cell_amount_not_on_borders) + ", " + \
    # #         "average pixel size_" + str(cell_pixel_size_average) + '.jpg', 
    # #     dpi=400, 
    # #     format='jpg', 
    # #     bbox_inches='tight'
    # # )
    # plt.show()

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
    for i in range(26, 27):
        path_cell_subfolder = "01"
        path_cell_name = "t" + format_name_digits(i) + ".tif"
        path_cell = str(image_src_path + path_divide + path_cell_subfolder + path_divide + path_cell_name)
        get_cell_segment_info(path_cell)
    
    '''
    What do those SEG and TRA look like?
    '''
    # path_seg = str(image_src_path + "\\01_GT\\SEG\\man_seg088.tif")
    # path_tra = str(image_src_path + "\\01_GT\\TRA\\man_track013.tif")
    # cell_seg = cv2.imread(path_seg, -1)
    # cell_tra = cv2.imread(path_tra, -1)

    # Check SEG
    # print(cell_seg.shape)
    # plt.title("cell_seg")
    # plt.imshow(cell_seg)
    # plt.show()

    # Check TRA
    # plt.title("cell_tra")
    # plt.imshow(cell_tra)
    # plt.show()
    