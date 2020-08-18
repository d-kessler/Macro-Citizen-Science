import openpyxl
import random
import os
from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, \
    Subject, User, Workflow


def configure_subject_set(subject_type):
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

    need_new_set = input('Would you like to create a new {} subject set? [yes/no]: '.format(subject_type))
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


def configure_excel(excel_file_path_):
    # TODO: uncomment

    wb = openpyxl.load_workbook(filename=excel_file_path_)
    ws = wb['Sheet1']
    for row in range(1, int(1e10)):
        if ws.cell(row, 1).value is None:
            first_empty_row = row
            break

    return wb, ws, first_empty_row


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