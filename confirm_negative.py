import os
import csv
from PIL import Image, ImageDraw, ExifTags
from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, \
    Subject, User, Workflow
import matplotlib.pyplot as plt

from config import *
from image_tools import *


def make_neg_pos_folders(image_folder_path_):
    """Create a folder for simulation images"""
    neg_folder_path = os.path.join(image_folder_path_, r"negatives")
    try:
        os.mkdir(neg_folder_path)
    except:
        clear_folder(neg_folder_path)

    pos_folder_path = os.path.join(image_folder_path_, r"positives")
    try:
        os.mkdir(pos_folder_path)
    except:
        clear_folder(pos_folder_path)

    return neg_folder_path, pos_folder_path


def configure_metadata():
    feedback_id = 'no_meltpatch'
    training_subject = 'true'

    return feedback_id, training_subject


def configure_csv(sim_folder_path_):
    """Configuring csv file, saving in simulation images folder"""
    csv_file_name = 'simulation_subjects.csv'
    csv_file_path = os.path.join(sim_folder_path_, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#training_subject', '#feedback_1_id', '#classification']
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)
        csv_writer.writeheader()

    return csv_file_name, csv_file_path, metadata_fields


def write_metadata_into_excel(ws, i, subject_id, neg_file_name, training_subject, feedback_id, classification):
    ws.cell(row=i, column=1).value = subject_id
    ws.cell(row=i, column=2).value = str(neg_file_name)
    ws.cell(row=i, column=3).value = training_subject
    ws.cell(row=i, column=4).value = feedback_id
    ws.cell(row=i, column=5).value = classification


def write_metadata_into_csv(csv_file_path_, metadata_fields_, subject_id, neg_file_name, training_subject, feedback_id,
                            classification):
    with open(csv_file_path_, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields_)

        row = {'!subject_id': subject_id}
        row['#file_name'] = str(neg_file_name)
        row['#training_subject'] = training_subject
        row['#feedback_1_id'] = feedback_id
        row['#classification'] = classification
        csv_writer.writerow(row)


def create_negs_from_process_images(folder_path, upload_now_, max_sample_, negative_set_id_=None):
    image_folder_path = folder_path
    upload_now = upload_now_
    negative_set_id = negative_set_id_
    max_sample = max_sample_

    create_negs(image_folder_path, upload_now, max_sample, negative_set_id_=negative_set_id)


def create_negs(image_folder_path, upload_now_, max_sample, negative_set_id_=None):
    # Create a folder for simulated images within the folder where images are fetched
    neg_folder_path, pos_folder_path = make_neg_pos_folders(image_folder_path)  # (specified folder with images)

    # Configure excel file to be written into, find the first empty row
    excel_file_path = r"manifests/Negative_Manifest.xlsx"
    wb, ws, first_empty_row = configure_excel(excel_file_path)

    # Fetch predefined metadata fields
    feedback_id, training_subject = configure_metadata()

    # Create csv file; get its name, file path, and metadata fields (column headers)
    csv_file_name, csv_file_path, metadata_fields = configure_csv(neg_folder_path)

    # Get list of file names in a directory
    file_names = get_file_names(image_folder_path)

    # Sample from the file names list at most 'max_sample' # of files, store in new list
    if len(file_names) >= max_sample:
        sample_number = max_sample
    else:
        sample_number = len(file_names)
    select_file_names = sample_from_file_names(file_names, sample_number)  # (list to sample from, number to sample)

    # Index variable used to assign subject ID, decide value of axesLength
    i = first_empty_row

    # Iterating through sampled files
    for filename in select_file_names:
        # Assigning a subject ID equal to 'n' (for negative) plus the total number of such subjects as of this one's addition
        subject_id = 'n' + str(i - 1)

        # Getting the file's full path, creating a path for the simulated image to be created from it
        image_file_path = os.path.join(image_folder_path, filename)
        neg_file_name = r"neg_" + filename
        neg_file_path = os.path.join(neg_folder_path, neg_file_name)

        # Creating an cv2 image instance named 'image'
        image = Image.open(image_file_path)

        print('Displaying image {} of {}'.format(select_file_names.index(filename) + 1, len(select_file_names)))

        plt.imshow(image)
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close()

        is_neg = input('Does this image contain a potential melt patch? [y/n]: ')

        if is_neg == 'n':
            classification = 'negative'
            image.save(neg_file_path)
        elif is_neg == 'y':
            classification = 'positive'
            pos_file_name = r"pos_" + filename
            pos_file_path = os.path.join(pos_folder_path, pos_file_name)

        # Write metadata values into both the specified excel file and created csv
        write_metadata_into_excel(ws, i, subject_id, neg_file_name, training_subject, feedback_id, classification)
        write_metadata_into_csv(csv_file_path, metadata_fields, subject_id, neg_file_name, training_subject,
                                feedback_id, classification)

        i += 1

    # Saving the excel manifest --- it must be closed
    wb.save(excel_file_path)

    # Executing terminal commands to upload images
    if upload_now_ == 'y':
        cmd = 'panoptes subject-set upload-subjects {} {}'.format(negative_set_id_, csv_file_path)
        os.system(cmd)
        print('\nNegative subjects uploaded.')