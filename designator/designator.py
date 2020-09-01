from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, \
    Subject, User, Workflow

# Logging in
Panoptes.connect(username='macrodarkmatter@gmail.com', password='2pP3pbKkUze2')

# Linking project, subject sets
project_id = 11726
macro_project = Project.find(project_id)
macro_project.save()

# configuring, fetching subject sets
experiment_subjects_id = 86450
experiment_subjects = SubjectSet.find(experiment_subjects_id)
experiment_subjects.save()

simulation_subjects_id = 86778
simulation_subjects = SubjectSet.find(simulation_subjects_id)
simulation_subjects.save()

negative_subjects_id = 86840
negative_subjects = SubjectSet.find(negative_subjects_id)
negative_subjects.save()

# Designator
workflow_id = 14437
workflow = Workflow.find(workflow_id)
workflow.configuration['training_set_ids'] = [simulation_subjects_id, negative_subjects_id]
workflow.configuration['training_chances'] = [[0.40] * 50, [0.20] * 50]
workflow.configuration['training_default_chances'] = [0.1]
workflow.configuration['subject_queue_page_size'] = 10  # determines how many subjects are loaded in queue at one time

# Panoptes retirement setting
workflow.retirement[
    'criteria'] = 'never_retire'  # training subjects not retired, experiment subjects retired via SWAP/Caesar

# Saving
workflow.modified_attributes.add('configuration')
workflow.save()
