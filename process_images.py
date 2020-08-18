import sys
import os
import io
import json
import csv
import image_slicer
import openpyxl
from PIL import Image, ImageDraw, ExifTags
from pathlib import Path
from datetime import datetime
from panoptes_client import Panoptes, Project, SubjectSet, Subject


def configure():
    """Get file & slab information, create new folder for images, configure excel & csv files"""
    global subject_set_id, image_folder_path, excel_file_path, warehouse_name, location, granite_type, slab_id, should_crop_into_four, \
        cropped_folder_path, processed_folder_path, csv_file_name, csv_file_path, metadata_fields, wb, ws, first_empty_row

    subject_set_id = 86450

    # # getting user input for metadata fields
    # image_folder_path = input('Enter the folder path: ')
    # excel_file_path = input('Excel file path: ')
    # warehouse_name = input('Warehouse name: ')
    # location = input('Location (City, State): ')
    # granite_type = input('Granite type: ')
    # slab_id = input('Slab ID: ')
    should_crop_into_four = input('Should the images be cropped into 4 parts? Enter \'yes\' or \'no\':  ')

    # image_folder_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 3 6x8 no flash"
    image_folder_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 1 6x8 no flash"
    excel_file_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Experiment_Manifest.xlsx"
    warehouse_name = 'United Stone International'
    location = 'Solon, Ohio'
    granite_type = 'Dallas White'
    slab_id = '1151|20'

    # creating a folder for cropped images
    cropped_folder_path = os.path.join(image_folder_path, r"cropped_" + os.path.basename(image_folder_path))
    try:
        os.mkdir(cropped_folder_path)
    except:
        print('Cropped Folder Exists.')

    # creating a folder for processed images
    processed_folder_path = os.path.join(image_folder_path, r"processed_" + os.path.basename(image_folder_path))
    try:
        os.mkdir(processed_folder_path)
    except:
        print('Processed Folder Exists.')

    # configuring excel file, finding first empty row
    wb = openpyxl.load_workbook(filename=excel_file_path)
    ws = wb['Sheet1']
    for row in range(1, int(1e10)):
        if ws.cell(row, 1).value is None:
            first_empty_row = row
            break

    # configuring csv file, saving in processed images folder
    csv_file_name = 'experiment_subjects_csv.csv'
    csv_file_path = os.path.join(processed_folder_path, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#warehouse', '#location', '#granite_type', '#slab_id',
                           '#date_time',
                           '#latitude_longitude']
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


def crop_into_four():
    global image_folder_path

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
            if j == 2:
                im = im.crop(section_2)
            if j == 3:
                im = im.crop(section_3)
            if j == 4:
                im = im.crop(section_4)

            reformatted_cropped_file_path = cropped_file_path.replace(str(extension),
                                                                      "_{}{}".format(str(j), str(extension)))

            im = im.save(reformatted_cropped_file_path, exif=image_exif_)

    get_file_names(cropped_folder_path)
    image_folder_path = cropped_folder_path


def resize_to_limit(image_file_path_, size_limit=600000):
    """Resize images to size_limit, save to new file"""

    img = Image.open(image_file_path_)
    img_exif = img.info['exif']
    aspect = img.size[0] / img.size[1]

    while True:
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            data = buffer.getvalue()
        filesize = len(data)
        size_deviation = filesize / size_limit
        # print("size: {}; factor: {:.3f}".format(filesize, size_deviation))

        if size_deviation <= 1:
            img.save(processed_file_path, exif=img_exif)
            break
        else:
            new_width = img.size[0] / (1 * (size_deviation ** 0.5))
            new_height = new_width / aspect

            img = img.resize((int(new_width), int(new_height)))


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

        csv_writer.writerow(row)


def draw_scale_bar(pil_file_, image_exif_):
    im = pil_file_
    pix_width, pix_height = im.size

    inch_to_mm_conversion = 25.4
    true_height = 6 * inch_to_mm_conversion
    true_width = 8 * inch_to_mm_conversion

    mm_per_pix = true_height / pix_height

    five_mm = 5 / mm_per_pix

    scale_bar_pix_length1 = 1 / mm_per_pix  # (1mm) / (mm/pix) = pix in 1mm
    scale_bar_pix_length2 = 2 / mm_per_pix
    scale_bar_pix_length3 = 3 / mm_per_pix
    scale_bar_pix_length4 = 4 / mm_per_pix
    scale_bar_pix_length5 = 5 / mm_per_pix
    scale_bar_pix_length10 = 10 / mm_per_pix

    scale_bar1_pix_coords = [(pix_width / 2 - 3 * five_mm, pix_height / 2),
                             (pix_width / 2 - 3 * five_mm,
                              pix_height / 2 + scale_bar_pix_length1)]  # [(start x, start y), (end x, end y)]
    scale_bar2_pix_coords = [(pix_width / 2 - 2 * five_mm, pix_height / 2),
                             (pix_width / 2 - 2 * five_mm, pix_height / 2 + scale_bar_pix_length2)]
    scale_bar3_pix_coords = [(pix_width / 2 - 1 * five_mm, pix_height / 2),
                             (pix_width / 2 - 1 * five_mm, pix_height / 2 + scale_bar_pix_length3)]
    scale_bar4_pix_coords = [(pix_width / 2, pix_height / 2),
                             (pix_width / 2, pix_height / 2 + scale_bar_pix_length4)]
    scale_bar5_pix_coords = [(pix_width / 2 + 1 * five_mm, pix_height / 2),
                             (pix_width / 2 + 1 * five_mm, pix_height / 2 + scale_bar_pix_length5)]
    scale_bar10_pix_coords = [(pix_width / 2 + 2 * five_mm, pix_height / 2),
                              (pix_width / 2 + 2 * five_mm, pix_height / 2 + scale_bar_pix_length10)]

    yellow = (234, 255, 5)  # (R, G, B)

    draw = ImageDraw.Draw(im)

    draw.line(scale_bar1_pix_coords, fill=yellow, width=10)
    draw.line(scale_bar2_pix_coords, fill=yellow, width=10)
    draw.line(scale_bar3_pix_coords, fill=yellow, width=10)
    draw.line(scale_bar4_pix_coords, fill=yellow, width=10)
    draw.line(scale_bar5_pix_coords, fill=yellow, width=10)
    draw.line(scale_bar10_pix_coords, fill=yellow, width=10)

    im.save(processed_file_path, exif=image_exif_)


configure()
get_file_names(image_folder_path)

# cropping images, saving to new folder (if needed)
if should_crop_into_four == 'yes':
    crop_into_four()

# resizing images, getting and filling metadata into excel file & csv
i = first_empty_row

for filename in file_names:
    # subject id to be written into excel & csv files
    subject_id = 'e' + str(i - 1)

    image_file_path = os.path.join(image_folder_path, filename)
    processed_file_name = r"proc_" + filename
    processed_file_path = os.path.join(processed_folder_path, processed_file_name)

    # creating PIL instance
    pil_file = Image.open(image_file_path)
    image_exif = pil_file.info['exif']

    # drawing scale bar on image
    draw_scale_bar(pil_file, image_exif)

    # resizing, saving to new folder
    resize_to_limit(processed_file_path)

    # getting image exif data
    exif = {ExifTags.TAGS[k]: v for k, v in pil_file._getexif().items() if k in ExifTags.TAGS}
    get_date(exif)
    get_gps(exif)

    # writing image information into excel & csv files
    # write_metadata_into_excel()
    write_metadata_into_csv()

    print('{} of {} completed.'.format((file_names.index(filename) + 1), len(file_names)))
    i += 1

# save excel manifest --- the excel file must be closed
wb.save(excel_file_path)

print(
    '\nTo upload subjects, \nEnter into the command line: \n    chdir {}\n\nFollowed by: \n    panoptes subject-set upload-subjects {} {}'.format(
        processed_folder_path, subject_set_id, csv_file_name))