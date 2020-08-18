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
import image_slicer
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
    aspect = img.size[0] / img.size[1]

    while True:
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            data = buffer.getvalue()
        filesize = len(data)
        size_deviation = filesize / size_limit
        # print("size: {}; factor: {:.3f}".format(filesize, size_deviation))

        if size_deviation <= 1:
            img.save(image_file_path_)
            break
        else:
            new_width = img.size[0] / (1 * (size_deviation ** 0.5))
            new_height = new_width / aspect

            img = img.resize((int(new_width), int(new_height)))