import sys
import json
import os
import numpy as np

from kswap import kSWAP
from offline_swap_config import Config
# from promote_subjects import PromoteSubjects


def run_swap(classifications_csv_path, golds_csv_path, workflow=16311, promote=True):
    """
    csv_paths = ('./CSVs/swap-gold-labels.csv', './CSVs/swap-classifications.csv')
    """
    # Retrieve swap configuration from 'offline_swap_config.py'
    swap_config = Config(workflow=workflow)
    # Create a kSWAP instance
    swap = kSWAP(config=swap_config)
    # Load subjects, users from 'offline_swap.db'
    swap = swap.load()
    # Run kSWAP on .csv files
    swap.run_offline(golds_csv_path, classifications_csv_path)
    # Save new subjects, users to 'offline_swap.db'
    swap.save()
    # Retrieve updated 'subjects', 'users' dictionaries from 'offline_swap.db'
    del swap
    swap = kSWAP(config=swap_config)
    swap = swap.load()

    # TODO: UNCOMMENT AND DELETE WHAT'S BELOW
    # # If promote=True, continue to identify subjects eligible for promotion
    # if promote is True:
    #     PromoteSubjects(swap, swap_config, classifications_csv_path)

    # Set positive probability threshold for a subject's promotion to the inspection workflow
    positive_prior = swap_config.p0['1']
    promotion_threshold = 1.5 * positive_prior
    """
    (kSWAP instance).users[(user ID)] is an instance of the 'User' class, having attributes:
        user_id, classes, k = len(classes), gamma, user_default, user_score, confusion_matrix, history
        history = [(subject ID), ('user_score')]
            Example: [201, {"0": [0.6, 0.4], "1": [0.3, 0.7]}]
    (kSWAP instance).subjects[(subject ID)] is an instance of the 'Subject' class, having attributes:
        subject_id, score, classes, gold_label, epsilon, retired (boolean), retired_as, seen, history
        history = [(classification ID), (user ID), ('user_score'), (submitted classification), (subject score)]
            Example: [1001, 101, {"0": [0.6, 0.4], "1": [0.3, 0.7]}, 1, {"0": 0.35, "1": 0.65}]
    """
    # Get lists of  'Subject' instances
    subjects = list(swap.subjects.values())
    print([sh[0] for sh in subjects[3].history])
    # # Get subjects' gold labels (-1: non-training, 0: negative, 1: positive)
    # subjects_gold_label = [subject.gold_label for subject in subjects]
    # # Get subjects' probabilities of being 'positive' (of containing a melt-patch)
    # subjects_positive_prob = [json.loads(subject.score)['1'] for subject in subjects]
    # # Zip subjects' IDs, gold labels, and positive probabilities in 'subject_tuples'
    # subject_dicts = [{'subject_instance': s, 'gold_label': g, 'positive_probability': p}
    #                  for (s, i, g, p) in zip(subjects, subjects_ids, subjects_gold_label, subjects_positive_prob)]
    # # Get annotation info of subjects whose positive probabilities surpass the 'promotion_threshold'
    # eligible_subjects = []
    # for subject in subjects:
    #     # Get subject's gold labels (-1: non-training, 0: negative, 1: positive)
    #     subject_id = subject.subject_id
    #     # Get subjects' probabilities of being 'positive' (of containing a melt-patch)
    #     gold_label = subject.gold_label
    #     positive_prob = json.loads(subject.score)['1']
    #     # Ignore training subjects
    #     if sd['gold_label'] in [0, 1]:
    #         continue
    #     # Out of the subjects that have passed the promotion threshold, get a list of the IDs the classifications where
    #     # a marking was made (which translates to a SWAP annotation value of 1)
    #     marking_classification_ids = []
    #     if sd['positive_probability'] > promotion_threshold:
    #         subject_history = sd['subject_instance'].history
    #         classifications = np.array([sh[-2] for sh in subject_history][1:])
    #         marking_indices = list(np.where(classifications == 1)[0])
    #         for mi in marking_indices:
    #             classification_id = subject_history[mi][0]
    #             if classification_id != '_':
    #                 marking_classification_ids.append(classification_id)
    #         eligible_subjects.append({'subject_id': sd['subject_id'], 'marking_classification_ids': marking_classification_ids})
    #     else:
    #         continue
    #     # ps = PromoteSubjects(eligible_subjects)
    # print('done')


def main():
    try:
        os.remove("db/offline_swap.db")
    except FileNotFoundError:
        pass
    csv_paths = ('./csv/swap-demo-gold-labels.csv', './csv/swap-demo-classifications.csv')
    run_swap(csv_paths, 0)


if __name__ == '__main__':
    main()
