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
import io
from PIL import Image, ImageDraw, ExifTags


def get_mm_per_pixel(img):
    """Getting the ratio of millimeters per pixel of image using fixed image dimensions"""
    image_dimensions = [6, 8]  # height, width of image in inches
    inch_to_mm_conversion = 25.4
    try:
        pix_height, pix_width, _ = img.shape
    except:
        pix_height, pix_width = img.size
    true_height = image_dimensions[0] * inch_to_mm_conversion
    true_width = image_dimensions[1] * inch_to_mm_conversion
    mm_per_pixel = true_height / pix_height

    return mm_per_pixel


def resize_to_limit(image_file_path_, size_limit=600000):
    """Resize images to size_limit"""

    img = Image.open(image_file_path_)
    image_exif = img.info['exif']
    aspect = img.size[0] / img.size[1]

    while True:
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            data = buffer.getvalue()
        filesize = len(data)
        size_deviation = filesize / size_limit
        # print("size: {}; factor: {:.3f}".format(filesize, size_deviation))

        if size_deviation <= 1:
            img.save(image_file_path_, exif=image_exif)
            break
        else:
            new_width = img.size[0] / (1 * (size_deviation ** 0.5))
            new_height = new_width / aspect

            img = img.resize((int(new_width), int(new_height)))


def draw_scale_bars(save_file_path, pil_file_, image_exif_, lower_limit_):
    """Draw a series of scale bars on the images of size equal to lower limit (for users' reference)"""

    im = pil_file_
    pix_width, pix_height = im.size

    mm_per_pixel = get_mm_per_pixel(im)

    # TODO: Clean up

    scale_bar_pix_length = lower_limit_ / mm_per_pixel

    if lower_limit_ >= 3:
        number_of_bars = 10
    elif lower_limit_ == 2:
        number_of_bars = 10
    elif lower_limit_ == 1:
        number_of_bars = 20

    side_center_of_region = (pix_height / number_of_bars) / 2
    top_center_of_region = (pix_width / number_of_bars) / 2

    draw = ImageDraw.Draw(im)
    color = (100, 255, 0)  # (R, G, B)
    side_start_coords = [15, side_center_of_region-(.5*scale_bar_pix_length)]
    side_end_coords = [side_start_coords[0], side_start_coords[1] + scale_bar_pix_length]

    top_start_coords = [top_center_of_region-(.5*scale_bar_pix_length), 15]
    top_end_coords = [top_start_coords[0] + scale_bar_pix_length, top_start_coords[1]]

    side_bar_pix_coords = [tuple(side_start_coords), tuple(side_end_coords)]  # [(start x, start y), (end x, end y)]
    top_bar_pix_coords = [tuple(top_start_coords), tuple(top_end_coords)]

    for j in range(1, number_of_bars+1):

        draw.line(side_bar_pix_coords, fill=color, width=10)
        draw.line(top_bar_pix_coords, fill=color, width=10)

        side_start_coords[1] += (2 * side_center_of_region)
        top_start_coords[0] += (2 * top_center_of_region)

        if j % 2 != 0:
            side_start_coords[0] = pix_width - side_start_coords[0]
            top_start_coords[1] = pix_height - top_start_coords[1]
        else:
            side_start_coords[0] = 15
            top_start_coords[1] = 15

        side_end_coords = [side_start_coords[0], side_start_coords[1] + scale_bar_pix_length]
        top_end_coords = [top_start_coords[0] + scale_bar_pix_length, top_start_coords[1]]

        side_bar_pix_coords = [tuple(side_start_coords), tuple(side_end_coords)]
        top_bar_pix_coords = [tuple(top_start_coords), tuple(top_end_coords)]

        # plt.imshow(im)
        # plt.show(block=False)
        # plt.waitforbuttonpress(0)
        # plt.close()

    # im.save(save_file_path, exif=image_exif_)
    im.save(save_file_path, exif=image_exif_)


def canny_thresholds(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5