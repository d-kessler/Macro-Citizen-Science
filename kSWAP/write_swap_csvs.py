import csv
import json
import os


class WriteCSVs:
    """
    Convert Zooniverse data export .csv file into a format amenable to (k)SWAP, write golds .csv.
    """
    def __init__(self, csv_file_path, starting_row, destination_folder_path, destination_file_name):
        """
        csv_file_path: Zooniverse data export .csv file path.
        starting_row: The row of the 'csv_file' at which to begin.
        destination_folder_path: Folder to save the .csv files.
        destination_file_name: File name given to the new classification .csv and prefixing the golds .csv.
        """
        self.csv_file_path = csv_file_path
        self.starting_row = starting_row
        self.destination_folder_path = destination_folder_path
        self.destination_file_name = destination_file_name

    @staticmethod
    def parse_csv(csv_file_path, starting_row):
        with open(csv_file_path, 'r') as read_csv_file:
            # Create csv reader instance
            csv_reader = csv.DictReader(read_csv_file)
            # Initialize lists
            metadata = []
            annotations = []
            subject_data = []
            rows = []
            # Skip up to specified starting row
            for i in range(starting_row - 2):
                next(csv_reader)
            # Iterate over rows, appending columns to lists
            for row in csv_reader:
                metadata.append(json.loads(row['metadata']))
                annotations.append(json.loads(row['annotations']))
                subject_data.append(json.loads(row['subject_data']))
                rows.append(row)
        return metadata, annotations, subject_data, rows

    @staticmethod
    def parse_metadata(metadata):
        success = []
        for i in range(len(metadata)):
            try:
                feedback = metadata[i]['feedback']
                task = list(feedback.keys())[0]
                success.append(feedback[task][0]['success'])
            except KeyError:
                success.append(None)
        return success

    @staticmethod
    def parse_annotations(annotations):
        user_cl = []
        for i in range(len(annotations)):
            if annotations[i][0]['value']:
                user_cl.append('Positive')
            else:
                user_cl.append('Negative')
        return user_cl

    @staticmethod
    def parse_subject_data(subject_data):
        subject_id = []
        gold_label = []
        for i in range(len(subject_data)):
            znv_subject_id = list(subject_data[i].keys())[0]
            subject_dict = subject_data[i][znv_subject_id]
            our_subject_id = subject_dict['!subject_id']
            if 'e' in our_subject_id:
                continue
            elif 'n' in our_subject_id:
                gold_label.append(0)
            elif 's' in our_subject_id:
                gold_label.append(1)
            else:
                print(f'error with subject_id {our_subject_id} on row {i} of subject_data')
            subject_id.append(znv_subject_id)
        return subject_id, gold_label

    @staticmethod
    def swap_value(user_cl, success):
        """
        Key (annotation 'value' field):
        (training)     correct negative: 0
            - counts towards the user's true negative rate
        (training)     correct positive: 1
             -counts towards the user's true positive  rate
        (training)     wrong   negative: 1
            - counts towards the user's false positive rate
        (training)     wrong   positive: 0
            - counts towards the user's false negative rate
                - the user missed the annotations; we cannot judge
                the falsity of any other annotations made, so this
                cannot be treated as a false positive (only a false
                negative with respect to the simulation).
        (non-training)         negative: 0
        (non-training)         positive: 1
        """
        if success is not None:
            if user_cl == 'Negative':
                if success is True:
                    swap_annotation_value = 0
                elif success is False:
                    swap_annotation_value = 1
            if user_cl == 'Positive':
                if success is True:
                    swap_annotation_value = 1
                elif success is False:
                    swap_annotation_value = 0
        else:
            if user_cl == 'Negative':
                swap_annotation_value = 0
            if user_cl == 'Positive':
                swap_annotation_value = 1
        return swap_annotation_value

    @staticmethod
    def write_csv(destination_folder_path, destination_file_name, rows):
        destination_file_path = os.path.join(destination_folder_path, destination_file_name)
        with open(destination_file_path, 'w', newline='') as write_csv_file:
            csv_writer = csv.DictWriter(write_csv_file, fieldnames=list(rows[0].keys()))
            csv_writer.writeheader()
            for i in range(len(rows)):
                row = rows[i]
                csv_writer.writerow(row)

    def cl_swap_convert(self, csv_file_path, starting_row, destination_folder_path, destination_file_name):
        """
        Convert Zooniverse data export .csv file into a format amenable to (k)SWAP, write golds .csv.

        csv_file_path: Zooniverse data export .csv file path.
        starting_row: The row of the 'csv_file' at which to begin.
        destination_folder_path: Folder to save the .csv files.
        destination_file_name: File name given to the new classification .csv and prefixing the golds .csv.
        """
        # Parse classifications csv and data fields therein
        metadata, annotations, subject_data, rows = self.parse_csv(csv_file_path, starting_row)
        user_cl = self.parse_annotations(annotations)
        success = self.parse_metadata(metadata)
        subject_id, gold_label = self.parse_subject_data(subject_data)
        # Overwrite 'value' field of annotations field to accord with swap
        for i in range(len(user_cl)):
            swap_annotation_value = self.swap_value(user_cl[i], success[i])
            annotations[i][0]["value"] = str(swap_annotation_value)
            rows[i]["annotations"] = json.dumps([annotations[i][0]])
        # Write swap-able classifications csv
        self.write_csv(destination_folder_path, destination_file_name, rows)
        # Write 'golds' csv
        golds_rows = []
        for i in range(len(subject_id)):
            golds_row = {'subject_id': subject_id[i], 'gold': gold_label[i]}
            golds_rows.append(golds_row)
        self.write_csv(destination_folder_path, destination_file_name + '_golds', golds_rows)

    def run(self):
        self.cl_swap_convert(self.csv_file_path, self.starting_row, self.destination_folder_path,
                             self.destination_file_name)
