import cv2
import numpy as np
import math
import random
from matplotlib import pyplot as plt
import os
import statistics
from scipy.spatial import distance
from skimage.util import img_as_ubyte
import imutils
from imutils import perspective
from imutils import contours

from config import *
from image_tools import *


def get_stats(list_, quant_name=''):
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


def draw_ellipse(minor_axis_length, mm_per_pixel_, file_path_):
    """Drawing ellipse on random file in folder with semi-major axis length equal to lower limit"""

    img = cv2.imread(file_path_)

    height, width, _ = img.shape

    minVal = int(min(img.flatten()))

    center_coordinates = (random.randint(10, width), random.randint(10, height))
    angle = random.randint(0, 360)
    startAngle = 0
    endAngle = 360

    minor_to_major_ratio = 2 / 3

    minor_axis = minor_axis_length / mm_per_pixel_
    major_axis = minor_axis / minor_to_major_ratio

    axesLength = [major_axis, minor_axis]
    axesRadii = tuple([int(i / 2) for i in axesLength])

    color = (minVal, minVal, minVal)

    img = cv2.ellipse(img, center_coordinates, axesRadii, angle, startAngle, endAngle, color, -1)
    img = cv2.circle(img, center_coordinates, int(4 * axesLength[1]), (0, 255, 0), 10)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_grain_size_grain_density_and_ellipse_lower_limit(image_folder_path_, file_names_):
    """Separate grains from backgrounds, perform edge and contour detection, """

    image_folder_path = image_folder_path_
    file_names = file_names_
    select_file_names = sample_from_file_names(file_names, len(file_names))

    # Initializing lists
    grain_densities = []
    mean_grain_sizes = []

    for file_name in select_file_names:

        image_file_path = os.path.join(image_folder_path, file_name)

        # Reading image
        orig_img = cv2.imread(image_file_path)
        orig_img = img_as_ubyte(orig_img)

        # Get millimeters per pixel of image ratio
        mm_per_pixel = get_mm_per_pixel(orig_img)

        # Converting to grayscale
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        # Blurring grayscale image
        blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

        # Segmenting blurred image to separate grains from background
        segmentation_thresh = 85
        thresh_img = cv2.threshold(blurred_img, segmentation_thresh, 255, cv2.THRESH_BINARY)[1]

        # Opening (erosion followed by dilation) segmented image to close holes within grains
        kernel = np.ones((9, 9), np.uint8)
        opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, 5)

        # Detecting edges
        lower_canny, upper_canny = canny_thresholds(opened_img)
        edges_img = cv2.Canny(opened_img, lower_canny, upper_canny)

        # Dilating to ensure that contour paths are continuous
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil_edges_img = cv2.dilate(edges_img, dil_kernel)

        # UNCOMMENT TO SAVE INTERMEDIATE STEP IMAGES TO 'edits' FOLDER
        # edits_folder_path = os.path.join(image_folder_path, 'edits')
        # try:
        #     os.mkdir(edits_folder_path)
        # except:
        #     pass
        # extension = os.path.splitext(file_name)[-1]
        # thresh_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_thresh' + extension))
        # cv2.imwrite(thresh_file_path, thresh_img)
        # opened_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_opened' + extension))
        # cv2.imwrite(opened_file_path, opened_img)
        # edges_file_path = os.path.join(edits_folder_path, file_name.replace(extension, '_edges' + extension))
        # cv2.imwrite(edges_file_path, edges_img)
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
        inch_to_mm_conversion = 25.4
        image_area = (8 * 6) * (inch_to_mm_conversion ** 2)

        # Get grain density (area of grains / area of image)
        density_of_grains = total_grain_area / image_area

        grain_densities.append(density_of_grains)

        number_of_grains, mean_grain_size, median_grain_size, grain_size_sigma, minimum_grain_size, maximum_grain_size \
            = get_stats(grain_sizes, 'grain size')

        mean_grain_sizes.append(mean_grain_size)

        print('{} of {} images\' grains analyzed.'.format((select_file_names.index(file_name) + 1), len(select_file_names)))

    mean_grain_density = statistics.mean(grain_densities)
    mean_grain_size = statistics.mean(mean_grain_sizes)

    print('\nThe mean grain density is {}.\n'
          'The mean grain size is {} mm.'.format(mean_grain_density, mean_grain_size))

    lower_limit = math.ceil(((1 - mean_grain_density) ** (1 / 2)) * 3)

    good = 'no'
    while good == 'no':
        random_file = sample_from_file_names(select_file_names, 1)[0]
        random_file_path = os.path.join(image_folder_path, random_file)

        print('\nThis ellipse is {} millimeters.'.format(lower_limit))
        draw_ellipse(lower_limit, mm_per_pixel, random_file_path)
        good = input('    Is this a good lower limit? [yes/no]: ')

        if good == 'yes':
            break
        if good == 'no':
            change = input('    Should the lower limit be larger of smaller? [larger/smaller]: ')
            if change == 'larger':
                lower_limit += 1
            elif change == 'smaller':
                lower_limit -= 1

    print('\nThe lower limit is {} millimeters.\n'.format(lower_limit))

    return mean_grain_size, mean_grain_density, lower_limit
