# importing cv2 
import cv2
import argparse
import random
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76
import image_slicer

import sys
import os
import io
import csv
import openpyxl
from PIL import Image, ExifTags


def configure_subject_set():
    Panoptes.connect(username='macrodarkmatter@gmail.com', password='2pP3pbKkUze2')

    project_id = 11726
    macro_project = Project.find(project_id)
    macro_project.save()

    workflow_id = 14437
    workflow = Workflow.find(workflow_id)
    workflow.save()

    print('The existing subject sets are:')
    for ss in macro_project.links.subject_sets:
        set_name = ss.display_name
        print(ss, set_name)

    experiment_subjects = SubjectSet.find(86450)
    print(int(str(experiment_subjects).split()[1].replace('>', '')))
    pause()
    experiment_subjects_id = [int(s) for s in str(experiment_subjects).split() if s.isdigit()]
    print(experiment_subjects_id)
    pause()

    need_new_set = input('Would you like to create a new subject set? [yes/no]: ')
    if need_new_set == 'no':
        subject_set_id = input('Enter the ID of the existing set you\'d like to upload to: ')
    elif need_new_set == 'yes':
        subject_set_name = input('Enter a name for the new subject set: ')

        subject_set = SubjectSet()
        subject_set.links.project = macro_project
        subject_set.display_name = subject_set_name
        subject_set.save()
        workflow.links.subject_sets.add(subject_set)
        workflow.save()

        subject_set_id = int(str(subject_set).split()[1].replace('>', ''))

    return subject_set_id


def configure_files():
    """Get image folder path, input for whether to crop images"""
    # TODO: uncomment

    # image_folder_path = input('Enter the folder path: ')
    image_folder_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 3 6x8 no flash"
    # image_folder_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 1 6x8 no flash"

    # creating a folder for simulation images
    sim_folder_path = os.path.join(image_folder_path, r"sim_" + os.path.basename(image_folder_path))
    try:
        os.mkdir(sim_folder_path)
    except:
        print('Simulations Folder Exists.')

    return image_folder_path, sim_folder_path


def configure_excel():
    # TODO: uncomment

    # excel_file_path = input('Excel file path: ')
    excel_file_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Experiment_Manifest.xlsx"

    wb = openpyxl.load_workbook(filename=excel_file_path)
    ws = wb['Sheet1']
    for row in range(1, int(1e10)):
        if ws.cell(row, 1).value is None:
            first_empty_row = row
            break

    return excel_file_path, wb, ws, first_empty_row


def configure_metadata():
    # warehouse_name = input('Warehouse name: ')
    # location = input('Location (City, State): ')
    # granite_type = input('Granite type: ')
    # slab_id = input('Slab ID: ')

    warehouse_name = 'United Stone International'
    location = 'Solon, Ohio'
    granite_type = 'Dallas White'
    slab_id = '1151|20'

    return warehouse_name, location, granite_type, slab_id


def configure_csv():
    csv_file_name = 'processed_subjects.csv'
    csv_file_path = os.path.join(processed_folder_path, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#warehouse', '#location', '#granite_type', '#slab_id',
                           '#date_time', '#latitude_longitude', '#mean_grain_size', '#mean_grain_density', '#lower_limit']
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)
        csv_writer.writeheader()

    return csv_file_name, csv_file_path, metadata_fields


def get_file_names(image_folder_path_):
    """Create a list of image files in given directory"""

    all_file_names = os.listdir(image_folder_path_)
    file_names_ = []
    for file in all_file_names:
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            file_names_.append(file)

    return file_names_


def configure():
    global subject_set_id, feedback_id, training_subject, excel_file_path, \
        csv_file_name, csv_file_path, metadata_fields, wb, ws, first_empty_row

    # excel_file_path = input('Excel file path: ')

    excel_file_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Simulation_Manifest.xlsx"


    feedback_id = 'meltpatch'
    training_subject = 'true'

    # configuring excel file, finding first empty row
    wb = openpyxl.load_workbook(filename=excel_file_path)
    ws = wb['Sheet1']
    for row in range(1, int(1e10)):
        if ws.cell(row, 1).value is None:
            first_empty_row = row
            break

    # configuring csv file, saving in simulation images folder
    csv_file_name = 'simulation_subjects_csv.csv'
    csv_file_path = os.path.join(sim_folder_path, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#training_subject', '#feedback_1_id', '#feedback_1_x',
                           '#feedback_1_y', '#feedback_1_toleranceA',
                           '#feedback_1_toleranceB', '#feedback_1_theta',
                           '#minor_to_major_ratio']
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)
        csv_writer.writeheader()


def get_file_names(image_folder_path_):
    """Create a list of image files in given directory"""
    global file_names

    all_file_names = os.listdir(image_folder_path_)
    file_names = []
    for file in all_file_names:
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            file_names.append(file)


def sample_from_file_names():
    """Samples from the files in a given directory a number of files (of the user's choosing) to create simulations from"""
    global select_file_names

    number_to_sample = int(input(
        'There are {} images in the folder.\nEnter the number of these you\'d like to create simulations from: '.format(
            len(file_names))))
    select_file_names = random.sample(file_names, number_to_sample)


def get_mm_per_pixel(image_file_path_):
    global mm_per_pixel

    im = Image.open(image_file_path_)
    pix_width, pix_height = im.size

    inch_to_mm_conversion = 25.4
    true_height = 6 * inch_to_mm_conversion
    true_width = 8 * inch_to_mm_conversion

    mm_per_pixel = true_height / pix_height


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


def write_metadata_into_excel():
    ws.cell(row=i, column=1).value = subject_id
    ws.cell(row=i, column=2).value = str(sim_file_name)
    ws.cell(row=i, column=3).value = training_subject
    ws.cell(row=i, column=4).value = feedback_id
    ws.cell(row=i, column=5).value = center_coordinates[0]
    ws.cell(row=i, column=6).value = center_coordinates[1]
    ws.cell(row=i, column=7).value = axesLength[0]
    ws.cell(row=i, column=8).value = axesLength[1]
    ws.cell(row=i, column=9).value = angle
    ws.cell(row=i, column=10).value = minor_to_major_ratio


def write_metadata_into_csv():
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)

        row = {'!subject_id': subject_id}
        row['#file_name'] = str(sim_file_name)
        row['#training_subject'] = training_subject
        row['#feedback_1_id'] = feedback_id
        row['#feedback_1_x'] = center_coordinates[0]
        row['#feedback_1_y'] = center_coordinates[1]
        row['#feedback_1_toleranceA'] = axesLength[0]
        row['#feedback_1_toleranceB'] = axesLength[1]
        row['#feedback_1_theta'] = angle
        row['#minor_to_major_ratio'] = minor_to_major_ratio
        csv_writer.writerow(row)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-r", "--radius", type=int,
                help="radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

configure()
get_file_names(image_folder_path)
sample_from_file_names()

subject_set_id = 86778
configure_subject_set

i = first_empty_row
for filename in select_file_names:
    # Assigning a subject ID equal to 's' (for simulation) plus the total number of such subjects as of this one's addition
    subject_id = 's' + str(i - 1)

    # Getting the file's full path, creating a path for the simulated image to be created from it
    image_file_path = os.path.join(image_folder_path, filename)
    sim_file_name = r"sim_" + filename
    sim_file_path = os.path.join(sim_folder_path, sim_file_name)

    # Path
    image = cv2.imread(image_file_path)

    # Reading an image in default mode getting width and height
    (height, width, channel) = image.shape

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    gray = cv2.blur(gray, (10, 10))

    # Find the darkest and brightest region
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # Setting ellipse parameters
    center_coordinates = (random.randint(10, width), random.randint(10, height))
    angle = random.randint(0, 360)
    startAngle = 0
    endAngle = 360

    # Finding the number of mm (on the image's 6x8 inch scale) per pixel
    get_mm_per_pixel(image_file_path)

    # Setting the ratio of the length of the major to minor axis to either 2/5, 3/5, or 4/5
    # minor_to_major_ratio = random.randint(2, 4) / 5
    minor_to_major_ratio = 2/3

    # Creating two minor axis sizes: 1mm, 3mm, and 5mm
    minor_axes = [1 / mm_per_pixel, 3 / mm_per_pixel, 5 / mm_per_pixel, 10 / mm_per_pixel]
    major_axes = [i / minor_to_major_ratio for i in minor_axes]

    # # Creating an equal number of ellipses of semi-major axis size 3mm and 5mm
    # if i % 2 == 0:
    #     axesLength = [int(major_axes[0]), int(minor_axes[0])]
    # elif i % 3 == 0:
    #     axesLength = [int(major_axes[1]), int(minor_axes[1])]
    # else:
    #     axesLength = [int(major_axes[2]), int(minor_axes[2])]

    axesLength = [int(major_axes[0]), int(minor_axes[0])]

    # csv.ellipse takes half axes lengths as an input
    axesRadii = tuple([int(i / 2) for i in axesLength])

    # Gets darkest color in the image
    color = (minVal, minVal, minVal)

    # Draw a ellipse with blue line borders of thickness of -1 px
    image = cv2.ellipse(image, center_coordinates, axesRadii, angle,
                        startAngle, endAngle, color, -1)

    image = cv2.circle(image, center_coordinates, 4 * axesLength[0], (0, 255, 0), 10)

    # Creating noise
    noise = np.random.normal(1000., 1000., (1000, 1000, 3))
    noise_file_path = os.path.join(sim_folder_path, 'gaussian_noise.png')
    granite = cv2.imwrite(noise_file_path, noise)

    # Saving simulated image
    cv2.imwrite(sim_file_path, image)

    # Window name in which image is displayed
    # window_name = 'Image'

    # Displaying the image
    # print(minVal)
    # print(maxVal)
    # cv2.waitKey(0)
    # cv2.imshow("Blur", gray)
    # cv2.imshow('False Ellipse', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    resize_to_limit(sim_file_path)

    # write_metadata_into_excel()
    write_metadata_into_csv()

    print('{} of {} completed.'.format((select_file_names.index(filename) + 1), len(select_file_names)))
    i += 1

# save excel manifest --- the excel file must be closed
wb.save(excel_file_path)

print(
    '\nTo upload subjects, \nEnter into the command line: \n    chdir {}\n\nFollowed by: \n    panoptes subject-set upload-subjects {} {}'.format(
        sim_folder_path, subject_set_id, csv_file_name))
