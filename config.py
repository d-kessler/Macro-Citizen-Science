import openpyxl
import random
import os
import stat
from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, \
    Subject, User, Workflow
import shutil


def configure_subject_set(subject_type):
    Panoptes.connect(username='macrodarkmatter@gmail.com', password='2pP3pbKkUze2')

    project_id = 11726
    macro_project = Project.find(project_id)
    macro_project.save()

    workflow_id = 14437
    workflow = Workflow.find(workflow_id)
    workflow.save()

    set_ids = []
    print('\nThe existing subject sets are:')
    for ss in macro_project.links.subject_sets:
        set_name = ss.display_name
        set_id = int(str(ss).split()[1].replace('>', ''))
        set_ids.append(set_id)
        print(ss, set_name)

    need_new_set = input('\nWould you like to create a new {} subject set? [y/n]: '.format(subject_type.upper()))
    if need_new_set == 'n':
        subject_set_id = input('    Enter the ID of the existing set you\'d like to upload to: ')
        while (int(subject_set_id) in set_ids) is False:
            subject_set_id = input('    This ID does not exist; please enter a new one: ')
    elif need_new_set == 'y':
        subject_set_name = input('    Enter a name for the new subject set: ')

        subject_set = SubjectSet()
        subject_set.links.project = macro_project
        subject_set.display_name = subject_set_name
        subject_set.save()
        workflow.links.subject_sets.add(subject_set)
        workflow.save()

        subject_set_id = int(str(subject_set).split()[1].replace('>', ''))

    return subject_set_id, need_new_set


def configure_excel(excel_file_path_):

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


def clear_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    except PermissionError or FileNotFoundError:
        input('\nPermissions error for {}, '
              '\nExit the file window and restart the program.'.format(folder_path))


def config_designator(simulation_subjects_id, negative_subjects_id):
    Panoptes.connect(username='macrodarkmatter@gmail.com', password='2pP3pbKkUze2')

    project_id = 11726
    macro_project = Project.find(project_id)
    macro_project.save()

    workflow_id = 14437
    workflow = Workflow.find(workflow_id)
    workflow.save()

    workflow.configuration['training_set_ids'] = [simulation_subjects_id, negative_subjects_id]
    workflow.configuration['training_chances'] = [[0.40] * 50, [0.20] * 50]
    workflow.configuration['training_default_chances'] = [0.1]
    workflow.configuration[
        'subject_queue_page_size'] = 10  # determines how many subjects are loaded in queue at one time

    # Training subjects are not retired, experiment subjects are retired via SWAP/Caesar
    workflow.retirement['criteria'] = 'never_retire'

    # Saving
    workflow.modified_attributes.add('configuration')
    workflow.save()

    print('\nDesignator configured.')
