import sys
import os
import io
import json
import csv
import openpyxl
from PIL import Image, ImageDraw, ExifTags
from pathlib import Path
from datetime import datetime, date
from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, \
    Subject, User, Workflow
import matplotlib.pyplot as plt
from git import Repo

from grain_size_density import *
from config import *
from image_tools import *
from sim_melt_patches import *
from confirm_negative import *

# ------------- #
lower_limit = 2
# ------------- #


def make_folders():
    """Get image folder path, input for whether to crop images"""
    should_crop_into_four = input('Should the images be cropped into 4 parts? [y/n]: ')

    processed_folder_path = os.path.join(image_folder_path, r"processed")
    try:
        os.mkdir(processed_folder_path)
    except FileExistsError:
        clear_folder(processed_folder_path)

    if should_crop_into_four == 'y':
        cropped_folder_path = os.path.join(image_folder_path, r"cropped")
        try:
            os.mkdir(cropped_folder_path)
        except FileExistsError:
            clear_folder(cropped_folder_path)
    elif should_crop_into_four == 'n':
        cropped_folder_path = ''

    return processed_folder_path, should_crop_into_four, cropped_folder_path


def configure_metadata():
    warehouse_name = input('Warehouse name: ')
    location = input('Location (City, State): ')
    granite_type = input('Granite type: ')
    slab_id = input('Slab ID: ')
    number_of_columns = input('Number of Columns: ')

    # warehouse_name = 'United Stone International'
    # location = 'Solon, Ohio'
    # granite_type = 'Dallas White'
    # slab_id = '1151|20'

    return warehouse_name, location, granite_type, slab_id, number_of_columns


def configure_csv():
    csv_file_name = 'processed_subjects.csv'
    csv_file_path = os.path.join(processed_folder_path, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#warehouse', '#location', '#granite_type', '#slab_id',
                           '#date_time', '#latitude_longitude', '#mean_grain_size', '#mean_grain_density',
                           '#number_of_columns']
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)
        csv_writer.writeheader()

    return csv_file_name, csv_file_path, metadata_fields


def crop_into_four():
    """If specified, crop images into four parts, save to cropped_folder"""
    global image_folder_path, file_names

    for filename_ in file_names:
        image_file_path_ = os.path.join(image_folder_path, filename_)
        cropped_file_path = os.path.join(cropped_folder_path, filename_)
        extension = os.path.splitext(filename_)[-1]

        pil_file_ = Image.open(image_file_path_)
        image_exif_ = pil_file_.info['exif']

        width, height = pil_file_.size

        half_width = width / 2
        half_height = height / 2

        # starting from top left (0,0) and moving clockwise
        # (left, top, right, bottom)
        section_1 = (0, 0, half_width, half_height)
        section_2 = (half_width, 0, width, half_height)
        section_3 = (half_width, half_height, width, height)
        section_4 = (0, half_height, half_width, height)

        for j in range(1, 5):
            im = pil_file_

            if j == 1:
                im = im.crop(section_1)
                location = 'TL'
            if j == 2:
                im = im.crop(section_2)
                location = 'TR'
            if j == 3:
                im = im.crop(section_3)
                location = 'BR'
            if j == 4:
                im = im.crop(section_4)
                location = 'BL'

            reformatted_cropped_file_path = cropped_file_path.replace(str(extension),
                                                                      "_{}{}".format(location, str(extension)))

            im.save(reformatted_cropped_file_path, exif=image_exif_)

    file_names = get_file_names(cropped_folder_path)

    image_folder_path = cropped_folder_path

    print('\nImages cropped.\n')


def threshold_glare():
    pass


def get_date(exif_dict):
    """Get date/time image exif data (if available)"""
    global date, has_date
    date = ''
    has_date = 0

    try:
        date = datetime.strptime(exif_dict['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
        date = json.dumps(date, default=str)
        has_date = 1

        return date, has_date

    except:
        pass


def get_gps(exif_dict):
    """Get GPS image exif data (if available)"""
    global latitude, longitude, has_gps
    latitude = []
    longitude = []
    has_gps = 0

    gps_dict = {}
    try:
        for key in exif_dict['GPSInfo'].keys():
            gps_tag = ExifTags.GPSTAGS.get(key)
            gps_dict[gps_tag] = exif_dict['GPSInfo'][key]

        latitude_raw = gps_dict.get('GPSLatitude')
        longitude_raw = gps_dict.get('GPSLongitude')

        lat_ref = gps_dict.get('GPSLatitudeRef')
        long_ref = gps_dict.get('GPSLongitudeRef')

        if lat_ref == "S":
            latitude = -abs(convert_to_degrees(latitude_raw))
        else:
            latitude = convert_to_degrees(latitude_raw)

        if long_ref == "W":
            longitude = -abs(convert_to_degrees(longitude_raw))
        else:
            longitude = convert_to_degrees(longitude_raw)

        has_gps = 1

        return latitude, longitude, has_gps

    except:
        pass


def convert_to_degrees(value):
    """Convert GPS exif data to latitude/longitude degree format"""
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    return d + (m / 60.0) + (s / 3600.0)


def write_metadata_into_excel():
    ws.cell(row=i, column=1).value = subject_id
    ws.cell(row=i, column=2).value = str(processed_file_name)
    ws.cell(row=i, column=3).value = warehouse_name
    ws.cell(row=i, column=4).value = location
    ws.cell(row=i, column=5).value = granite_type
    ws.cell(row=i, column=6).value = slab_id
    if has_date == 1:
        ws.cell(row=i, column=7).value = date.replace("\"", "")
    elif has_date == 0:
        ws.cell(row=i, column=7).value = '-'
    if has_gps == 1:
        ws.cell(row=i, column=8).value = str(latitude) + ', ' + str(longitude)
    elif has_gps == 0:
        ws.cell(row=i, column=8).value = '-'
    ws.cell(row=i, column=9).value = mean_grain_size
    ws.cell(row=i, column=10).value = mean_grain_density
    ws.cell(row=i, column=11).value = number_of_columns


def write_metadata_into_csv():
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)

        row = {'!subject_id': subject_id}
        row['#file_name'] = str(processed_file_name)
        row['#warehouse'] = warehouse_name
        row['#location'] = location
        row['#granite_type'] = granite_type
        row['#slab_id'] = slab_id
        if has_date == 1:
            row['#date_time'] = date.replace("\"", "")
        elif has_date == 0:
            row['#date_time'] = '-'
        if has_gps == 1:
            row['#latitude_longitude'] = str(latitude) + ', ' + str(longitude)
        elif has_gps == 0:
            row['#latitude_longitude'] = '-'
        row['#mean_grain_size'] = mean_grain_size
        row['#mean_grain_density'] = mean_grain_density
        row['#number_of_columns'] = number_of_columns

        csv_writer.writerow(row)


def update_manifests():
    """Commit, push updated manifests to GitHub"""
    repo_dir = '.'
    repo = Repo(repo_dir)
    files_to_push = [r"manifests/Experiment_Manifest.xlsx", r"manifests/Simulation_Manifest.xlsx",
                     r"manifests/Negative_Manifest.xlsx"]
    commit_message = 'update manifests, {}'.format(date.today())
    repo.index.add(files_to_push)
    repo.index.commit(commit_message)
    origin = repo.remote('origin')
    origin.push()

    print('\nManifests pushed.')


def pause():
    input('Pause ')


# Configure subject set(s) to upload to
upload_now = input('Are you looking to upload these subjects now? [y/n]: ')
if upload_now == 'y':
    experiment_set_id, need_new_exp = configure_subject_set('experiment')
    simulation_set_id, need_new_sim = configure_subject_set('simulation')
    negative_set_id, need_new_neg = configure_subject_set('negative')
else:
    experiment_set_id = None
    simulation_set_id = None
    negative_set_id = None

# Get a list of subfolders in parent "images" folder
parent_folder = "images"
subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]

# Iterates through list of subfolders
for subfolder in subfolders:
    print('\nEntering into {}: folder {} of {}.'.format(subfolder, subfolders.index(subfolder) + 1, len(subfolders)))
    image_folder_path = os.path.join(parent_folder, subfolder)

    # Make requisite folders
    processed_folder_path, should_crop_into_four, cropped_folder_path = make_folders()

    # Configure excel
    excel_file_path = r"manifests/Experiment_Manifest.xlsx"
    wb, ws, first_empty_row = configure_excel(excel_file_path)

    warehouse_name, location, granite_type, slab_id, number_of_columns = configure_metadata()
    csv_file_name, csv_file_path, metadata_fields = configure_csv()

    file_names = get_file_names(image_folder_path)

    # Cropping images, saving to new folder (if specified)
    if should_crop_into_four == 'y':
        crop_into_four()

    # TODO: Move into main loop
    # Get grain size/density from images in subdirectory or cropped folder path
    mean_grain_size, mean_grain_density = get_grain_size_grain_density_and_ellipse_lower_limit(
        image_folder_path, file_names)

    # -------------------------------------------------------------------
    # Resizing images, getting and filling metadata into excel file & csv
    # -------------------------------------------------------------------

    # Initializing index variable, i
    i = first_empty_row

    # Initializing images' row and column location on the slab
    row = 1
    col = 0
    crop_count = 0

    for filename in file_names:
        # Assigning a subject ID equal to 'e' (for experiment) plus the total number of such subjects as of this one's addition
        subject_id = 'e' + str(i - 1)

        # Getting row number
        row_changed = 0
        if should_crop_into_four == 'y' and number_of_columns:
            if (file_names.index(filename)/4) % int(number_of_columns) == 0 and file_names.index(filename) != 0:
                row += 1
                row_changed = 1
        elif should_crop_into_four == 'n':
            if file_names.index(filename) % int(number_of_columns) == 0 and file_names.index(filename) != 0:
                row += 1
                row_changed = 1

        # Getting column number
        if row % 2 == 0:
            if should_crop_into_four == 'y':
                if crop_count % 4 == 0 and row_changed != 1:
                    col -= 1
            elif should_crop_into_four == 'n' and row_changed != 1:
                col -= 1
        elif row % 2 != 0:
            if should_crop_into_four == 'y' and row_changed != 1:
                if crop_count % 4 == 0:
                    col += 1
            elif should_crop_into_four == 'n' and row_changed != 1:
                col += 1
        else:
            row = 0

        image_file_path = os.path.join(image_folder_path, filename)
        extension = os.path.splitext(filename)[-1]
        if should_crop_into_four == 'y':
            crop_file_name = os.path.splitext(filename)[-2]
            crop_location = crop_file_name.split("_")[-1]
            processed_file_name = subject_id + r"_" + str(slab_id) + r"_" + str(row) + r"_" + str(col) + r"_" + \
                                  crop_location + str(extension)
        elif should_crop_into_four == 'n':
            processed_file_name = subject_id + r"_" + str(slab_id) + r"_" + str(row) + r"_" + str(col) + str(extension)
        processed_file_path = os.path.join(processed_folder_path, processed_file_name)

        # Creating PIL instance
        pil_file = Image.open(image_file_path)
        image_exif = pil_file.info['exif']

        # Threshold glare
        # TBD

        # Drawing scale bars on image
        draw_scale_bars(processed_file_path, pil_file, image_exif, lower_limit)

        # Resizing, saving to new folder
        resize_to_limit(processed_file_path)

        # Getting image exif data
        exif = {ExifTags.TAGS[k]: v for k, v in pil_file._getexif().items() if k in ExifTags.TAGS}
        get_date(exif)
        get_gps(exif)

        # Writing image information into excel & csv files
        # write_metadata_into_excel() # TODO: uncomment
        write_metadata_into_csv()

        print('\r{} of {} images processed.'.format((file_names.index(filename) + 1), len(file_names)), end="")
        i += 1
        crop_count += 1

    # Save excel manifest --- Note: the excel file must be closed
    wb.save(excel_file_path)

    # Executing terminal commands to upload images
    if upload_now == 'y':
        cmd = 'panoptes subject-set upload-subjects {} {}'.format(experiment_set_id, csv_file_path)
        os.system(cmd)
        print('\nExperiment subjects uploaded.')

    # TODO: delete
    make_sims = input('\nPress enter to begin creating simulations...')

    # Number of sims/negs created per folder; 5 for beta, 3 otherwise
    max_sample = 5

    # Running sim_melt_patches.py (script to make simulation images)
    sim_file_paths = create_sims_from_process_images(processed_folder_path, upload_now, lower_limit, max_sample,
                                                     simulation_set_id_=simulation_set_id)

    # TODO: delete
    make_negs = input('\nPress enter to begin creating confirmed negatives (NOTE: to close the window, press any key --- do not X-out.)...')

    # Running confirm_negative.py (script to make confirmed negative images)
    create_negs_from_process_images(processed_folder_path, upload_now, max_sample, negative_set_id_=negative_set_id)

# Configure designator for new subject set
if upload_now == 'y':
    if need_new_sim == 'y' or need_new_neg == 'y':
        config_designator(simulation_set_id, negative_set_id)

# TODO: delete
# Update manifests on GitHub
input('\nPress enter to push manifests...')
update_manifests()

# TODO: Make automatic
print('\nProcessing completed. Go to project builder if you\'d like to update the active subject sets.')

# set active subject sets
# simulations appearance
# calculate area lost to glare (& large dark patches?)
# have simulations avoid glare areas
# for beta: create more images via inversion/rotation

