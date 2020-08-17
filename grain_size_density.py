import cv2
import numpy as np
import math
import random
from matplotlib import pyplot as plt
import os
from PIL import Image, ExifTags
import skimage
import statistics
import scipy
from scipy.spatial import distance
from skimage.util import img_as_ubyte
import imutils
from imutils import perspective
from imutils import contours


def configure():
    """Configuring image folder path"""

    image_folder_path_ = input('Enter the image folder path: ')
    # image_folder_path_ = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 3 6x8 no flash - Copy"
    # image_folder_path_ = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 3 6x8 no flash - Copy"
    # image_folder_path_ = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 1 12x16 no flash - Copy\cropped_Slab 1 12x16 no flash - Copy"

    return image_folder_path_


def get_file_names(image_folder_path_):
    """Create a list of image files in given directory"""

    all_file_names = os.listdir(image_folder_path_)
    file_names_ = []
    for file in all_file_names:
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            file_names_.append(file)

    return file_names_


def sample_from_file_names(file_names_, number):
    """Samples from the files in a given directory"""

    select_file_names_ = random.sample(file_names_, number)

    return select_file_names_


def get_stats(list_, quant_name):
    number = len(list_)
    mean = statistics.mean(list_)
    median = statistics.median(list_)
    standard_deviation = statistics.stdev(list_)
    minimum = min(list_)
    maximum = max(list_)
    twenty_fifth = np.percentile(list_, 25)
    seventy_fifth = np.percentile(list_, 75)

    return number, mean, median, standard_deviation, minimum, maximum

    # print('There are {} {}s.\nThe mean is {}. \nThe median is {}'
    #       '\nThe standard deviation is {}.\nThe range is [{}, {}].\n'
    #       'The 25th and 75th percentiles are [{}, {}]'
    #       .format(number, quant_name, average, median, standard_deviation,
    #               minimum, maximum, twenty_fifth, seventy_fifth))


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def canny_thresholds(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper


def pause():
    input('pause')


def draw_ellipse(minor_axis_length):
    random_file = sample_from_file_names(select_file_names, 1)[0]
    random_file_path = os.path.join(image_folder_path, random_file)
    print(random_file_path)
    img = cv2.imread(random_file_path)

    height, width, _ = img.shape

    minVal = int(min(img.flatten()))

    center_coordinates = (random.randint(10, width), random.randint(10, height))
    angle = random.randint(0, 360)
    startAngle = 0
    endAngle = 360

    minor_to_major_ratio = 2 / 3

    minor_axis = minor_axis_length / mm_per_pixel
    major_axis = minor_axis / minor_to_major_ratio

    axesLength = [major_axis, minor_axis]
    axesRadii = tuple([int(i / 2) for i in axesLength])

    color = (minVal, minVal, minVal)

    img = cv2.ellipse(img, center_coordinates, axesRadii, angle, startAngle, endAngle, color, -1)
    img = cv2.circle(img, center_coordinates, int(4 * axesLength[1]), (0, 255, 0), 10)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


image_folder_path = configure()
file_names = get_file_names(image_folder_path)
select_file_names = sample_from_file_names(file_names, len(file_names))

# Initializing lists
grain_densities = []
mean_grain_sizes = []

for file_name in select_file_names:

    image_file_path = os.path.join(image_folder_path, file_name)

    # edits_folder_path = os.path.join(image_folder_path, 'edits')
    # try:
    #     os.mkdir(edits_folder_path)
    # except:
    #     pass
    # extension = os.path.splitext(file_name)[-1]

    # Reading image
    orig_img = cv2.imread(image_file_path)
    orig_img = img_as_ubyte(orig_img)

    # Converting to grayscale
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Getting the ratio of millimeters per pixel of image using fixed image dimensions
    image_dimensions = [6, 8]  # height, width in inches
    inch_to_mm_conversion = 25.4
    pix_height, pix_width = gray_img.shape
    true_height = image_dimensions[0] * inch_to_mm_conversion
    true_width = image_dimensions[1] * inch_to_mm_conversion
    mm_per_pixel = true_height / pix_height

    # Blurring grayscale image
    blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

    # Segmenting blurred image to separate grains from background
    segmentation_thresh = 85
    thresh_img = cv2.threshold(blurred_img, segmentation_thresh, 255, cv2.THRESH_BINARY)[1]

    # thresh_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_thresh' + extension))
    # cv2.imwrite(thresh_file_path, thresh_img)

    # Opening (erosion followed by dilation) segmented image to close holes within grains
    kernel = np.ones((9, 9), np.uint8)
    opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, 5)

    # opened_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_opened' + extension))
    # cv2.imwrite(opened_file_path, opened_img)

    # Detecting edges
    lower_canny, upper_canny = canny_thresholds(opened_img)
    edges_img = cv2.Canny(opened_img, lower_canny, upper_canny)

    # edges_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_edges' + extension))
    # cv2.imwrite(edges_file_path, edges_img)

    # Dilating to ensure that contour paths are continuous
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil_edges_img = cv2.dilate(edges_img, dil_kernel)

    # dil_edges_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_edges_dil' + extension))
    # cv2.imwrite(dil_edges_file_path, dil_edges_img)

    # Getting, sorting a list of contours
    contours_from_img = dil_edges_img
    contours_list = cv2.findContours(contours_from_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = imutils.grab_contours(contours_list)
    (contours_list, _) = contours.sort_contours(contours_list)

    # Initializing lists
    grain_contours = []
    grain_sizes = []

    ignored = []
    ig_grain_sizes = []

    center_xs = []
    center_ys = []

    for cont in contours_list:

        #  Creating bounding box
        box = cv2.minAreaRect(cont)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype='int')
        box = perspective.order_points(box)

        # Getting box vertices coordinates
        (top_left, top_right, bot_right, bot_left) = box

        # Getting box side midpoints
        (top_mid_x, top_mid_y) = midpoint(top_left, top_right)
        (bot_mid_x, bot_mid_y) = midpoint(bot_left, bot_right)
        (left_mid_x, left_mid_y) = midpoint(top_left, bot_left)
        (right_mid_x, right_mid_y) = midpoint(top_right, bot_right)

        # Getting box center coordinates
        center_x = np.average(box[:, 0])
        center_y = np.average(box[:, 1])
        center_xs.append(center_x)
        center_ys.append(center_y)

        # Measuring the semi-major and -minor axes of the box in pixels
        pix_height = distance.euclidean((top_mid_x, top_mid_y), (bot_mid_x, bot_mid_y))
        pix_width = distance.euclidean((left_mid_x, left_mid_y), (right_mid_x, right_mid_y))

        # Converting axes lengths to millimeters
        mm_height = pix_height * mm_per_pixel
        mm_width = pix_width * mm_per_pixel

        # Setting grain size equal to the average of the contour's axes lengths
        grain_size = (mm_height + mm_width) / 2

        # Ignoring specs that are less 0.5mm across (a value found after experimentation)
        if grain_size < 0.5:
            ig_grain_sizes.append(grain_size)
            ignored.append(cont)
            continue

        grain_contours.append(cont)
        grain_sizes.append(grain_size)

    # # Coloring in grains found by the program to check for completeness
    # draw_conts_img = cv2.cvtColor(contours_from_img, cv2.COLOR_GRAY2RGB)
    # colored_grains_img = cv2.drawContours(draw_conts_img, grain_contours, contourIdx=-1, color=(0, 255, 0),
    #                                       thickness=cv2.FILLED)  # grains counted are colored green
    # colored_grains_file_path = os.path.join(edits_folder_path,
    #                                         file_name.replace(extension, '_colored_grains' + extension))
    # cv2.imwrite(colored_grains_file_path, colored_grains_img)

    # Get list of grain areas in square millimeters
    grain_areas = []
    for cont in grain_contours:
        area = cv2.contourArea(cont) * mm_per_pixel ** 2
        grain_areas.append(area)
    total_grain_area = sum(grain_areas)
    image_area = (8 * 6) * (inch_to_mm_conversion ** 2)

    # Get grain density (area of grains / area of image)
    density_of_grains = total_grain_area / image_area

    grain_densities.append(density_of_grains)

    # print('The density of grains is {} (area of grain)/(area of image).\n'.format(density_of_grains))

    number_of_grains, mean_grain_size, median_grain_size, grain_size_sigma, minimum_grain_size, maximum_grain_size \
        = get_stats(grain_sizes, 'grain size')

    mean_grain_sizes.append(mean_grain_size)

    print('{} of {} completed.'.format((select_file_names.index(file_name) + 1), len(select_file_names)))

mean_grain_density = statistics.mean(grain_densities)
mean_grain_size = statistics.mean(mean_grain_sizes)

print('The mean grain density is {}.\n'
      'The mean grain size is {} mm.'.format(mean_grain_density, mean_grain_size))

lower_limit = math.ceil(((1 - mean_grain_density) ** (1 / 2)) * 3)

good = 'no'
while good == 'no':
    draw_ellipse(lower_limit)
    print('This ellipse is {} millimeters.'.format(lower_limit))
    good = input('Is this a good lower limit? [yes/no]: ')
    if good == 'yes':
        break
    lower_limit += 1

print('The lower limit is {} millimeters.'.format(lower_limit))
