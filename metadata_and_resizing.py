import sys
import os
import io
import json
import csv
import openpyxl
from PIL import Image, ExifTags
from pathlib import Path
from datetime import datetime
from panoptes_client import Panoptes, Project, SubjectSet, Subject


def configure():
    """Get file & slab information, create new folder for images, configure excel & csv files"""
    global image_folder_path, excel_file_path, warehouse_name, granite_type, slab_id, \
        resized_folder_path, csv_file_name, csv_file_path, metadata_fields, wb, ws, first_empty_row

    # image_folder_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Slab 3 6x8 no flash"
    # excel_file_path = r"C:\Users\dkess\OneDrive\Documents\CWRU\Macro\Data, Analysis\Experiment_Manifest.xlsx"
    # warehouse_name = 'United Stone International'
    # granite_type = 'Dallas White'
    # slab_id = '1151|20'

    # getting user input for metadata fields
    image_folder_path = input('Enter the folder path: ')
    excel_file_path = input('Excel file path: ')
    warehouse_name = input('Warehouse name: ')
    granite_type = input('Granite type: ')
    slab_id = input('Slab ID: ')

    # creating a folder for resized images
    resized_folder_path = os.path.join(image_folder_path, r"resized-" + os.path.basename(image_folder_path))
    try:
        os.mkdir(resized_folder_path)
    except:
        print('Resized Folder Exists.')

    # configuring excel file, finding first empty row
    wb = openpyxl.load_workbook(filename=excel_file_path)
    ws = wb['Sheet1']
    for cell in ws["B"]:
        if cell.value is None:
            first_empty_row = cell.row
            break

    # configuring csv file, saving in resized images folder
    csv_file_name = 'experiment_subjects_csv.csv'
    csv_file_path = os.path.join(resized_folder_path, csv_file_name)

    with open(csv_file_path, 'w', newline='') as f:
        metadata_fields = ['!subject_id', '#file_name', '#warehouse', '#granite_type', '#slab_id', '#date_time',
                           '#latitude_longitude']
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)
        csv_writer.writeheader()


def get_file_names():
    """Create a list of image files in given directory"""
    global file_names

    all_file_names = os.listdir(image_folder_path)
    file_names = []
    for file in all_file_names:
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
            file_names.append(file)


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


def resize_to_limit(image_file_path_, resized_file_path_, size_limit=600000):
    """Resize images to size_limit, save to new file"""
    img = img_orig = Image.open(image_file_path_)
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
            img.save(resized_file_path_, exif=img_exif)
            # print(img.size[0], img.size[1])
            break
        else:
            new_width = img.size[0] / (1 * (size_deviation ** 0.5))
            new_height = new_width / aspect

            img = img_orig.resize((int(new_width), int(new_height)))


def write_metadata_into_excel():
    
    ws.cell(row=i, column=1).value = subject_id
    ws.cell(row=i, column=2).value = str(resized_file_name)
    ws.cell(row=i, column=3).value = warehouse_name
    ws.cell(row=i, column=4).value = granite_type
    ws.cell(row=i, column=5).value = slab_id
    if has_date == 1:
        ws.cell(row=i, column=6).value = date.replace("\"", "")
    elif has_date == 0:
        ws.cell(row=i, column=6).value = '-'
    if has_gps == 1:
        ws.cell(row=i, column=7).value = str(latitude) + ', ' + str(longitude)
    elif has_gps == 0:
        ws.cell(row=i, column=7).value = '-'


def write_metadata_into_csv():
    
    with open(csv_file_path, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=metadata_fields)

        row = {'!subject_id': subject_id}
        row['#file_name'] = str(resized_file_name)
        row['#warehouse'] = warehouse_name
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


configure()
get_file_names()

# resize images, get and fill metadata into excel file & csv
i = first_empty_row
for filename in file_names:
    
    subject_id = 'e' + str(i - 1)
    resized_file_name = r"res-" + filename

    image_file_path = os.path.join(image_folder_path, filename)
    resized_file_path = os.path.join(resized_folder_path, resized_file_name)

    resize_to_limit(image_file_path, resized_file_path)

    # getting image exif data
    pil_file = Image.open(image_file_path)
    exif = {ExifTags.TAGS[k]: v for k, v in pil_file._getexif().items() if k in ExifTags.TAGS}

    get_date(exif)

    get_gps(exif)

    write_metadata_into_excel()
    write_metadata_into_csv()

    print('{} of {} completed.'.format((file_names.index(filename) + 1), len(file_names)))

    i += 1

# save excel manifest --- the excel file must be closed
wb.save(excel_file_path)

print(
    '\nTo upload subjects, \nEnter into the command line: \n    chdir {}\n\nFollowed by: \n    panoptes subject-set upload-subjects 86450 {}'.format(
        resized_folder_path, csv_file_name))
