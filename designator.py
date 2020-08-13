from panoptes_client import Panoptes, Classification, Collection, Project, ProjectPreferences, ProjectRole, SubjectSet, Subject, User, Workflow

# Logging in
Panoptes.connect(username='macrodarkmatter@gmail.com', password='2pP3pbKkUze2')

# Linking project, subject sets
project_id = 11726
macro_project = Project.find(project_id)
macro_project.save()

experiment_subjects_id = 86450
experiment_subjects = SubjectSet.find(experiment_subjects_id)
experiment_subjects.save()

training_subjects_id = 85478
training_subjects = SubjectSet.find(training_subjects_id)
training_subjects.save()

# Designator
workflow_id = 14437
workflow = Workflow.find(workflow_id)
workflow.configuration['training_set_ids'] = training_subjects_id
workflow.configuration['training_chances'] = [[0.40] * 50, [0.20] * 50]
workflow.configuration['training_default_chances'] = [0.1]
# workflow.configuration['subject_queue_page_size'] = ???

# Panoptes retirement setting
workflow.retirement['criteria'] = 'never_retire' # training subjects not retired, experiment/test subjects retired via SWAP/Caesar

# Saving
workflow.modified_attributes.add('configuration')
workflow.save()


