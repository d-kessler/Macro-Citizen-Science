import os
import csv
import json
import sqlite3

from collections import Counter


class User(object):
    def __init__(self,
                 user_id,
                 classes,
                 gamma,
                 user_default=None):
        self.user_id = user_id
        self.classes = classes
        self.k = len(classes)
        self.gamma = gamma
        if user_default:
            self.user_score = user_default
        else:
            self.initialise_user_score()
        self.initialise_confusion_matrix()
        self.history = [('_', self.user_score)]

    def initialise_confusion_matrix(self):
        self.confusion_matrix = {'matrix': [[0] * self.k for i in range(self.k)],
                                 'n_gold': [0] * self.k
                                 }

    def initialise_user_score(self):
        self.user_score = {}
        for i in range(self.k):
            self.user_score[self.classes[i]] = [1 / float(self.k)] * self.k

    def update_confusion_matrix(self, gold_label, label):
        confusion_matrix = self.confusion_matrix
        confusion_matrix['n_gold'][gold_label] += 1
        confusion_matrix['matrix'][gold_label][label] += 1
        self.confusion_matrix = confusion_matrix

    def update_user_score(self, gold_label, label):
        self.update_confusion_matrix(gold_label, label)
        user_score = {}
        for i, c_i in enumerate(self.classes):
            user_score[c_i] = [None] * len(self.classes)
            for j, c_j in enumerate(self.classes):
                user_score[c_i][j] = (self.confusion_matrix['matrix'][i][j] + self.gamma) \
                                     / (self.confusion_matrix['n_gold'][i] + 2.0 * self.gamma)
        self.user_score = user_score

    def dump(self):
        return (self.user_id,
                json.dumps(self.user_score),
                json.dumps(self.confusion_matrix),
                json.dumps(self.history))


class Subject(object):
    def __init__(self,
                 subject_id,
                 p0,
                 classes,
                 gold_label=-1,
                 epsilon=1e-9):
        '''
            p0 = a kx1 dimensional list of classwise priors.
        '''
        assert len(p0) == len(classes)
        self.subject_id = subject_id
        self.score = p0
        self.classes = classes
        self.k = len(self.classes)
        self.gold_label = gold_label
        self.epsilon = epsilon
        self.retired = False
        self.retired_as = None
        self.seen = 0
        self.history = [('_', '_', '_', '_', self.score)]

    def update_score(self, id, label, user):
        score = {c: None for c in self.classes}
        # print(self.score)
        # print(user.user_score)
        # print(label)
        if type(self.score) is str:
            self.score = json.loads(self.score)
        denomenator = sum([user.user_score[c][label] * self.score[c] for c in self.classes])

        for c in self.classes:
            numerator = (self.score[c] * user.user_score[c][label])
            score[c] = numerator / (denomenator + self.epsilon)

        self.score = score
        self.history.append((id, user.user_id, user.user_score, label, self.score))
        self.seen += 1

    def dump(self):
        return (self.subject_id,
                self.gold_label,
                json.dumps(self.score),
                self.retired,
                self.retired_as,
                self.seen,
                json.dumps(self.history))


class Classification(object):
    def __init__(self,
                 id,
                 user_id,
                 subject_id,
                 annotation,
                 label_map):

        self.id = int(id)
        try:
            self.user_id = int(user_id)
        except ValueError:
            self.user_id = user_id
        self.subject_id = int(subject_id)
        self.label_map = label_map
        self.label = self.parse(annotation)

    def parse(self, annotation):
        value = annotation[0]['value']
        if value in self.label_map.keys():
            return self.label_map[value]
        else:
            raise ValueError('Annotation value {} not recognised.'.format(value))


class kSWAP(object):
    def __init__(self,
                 config=None,
                 timeout=10):
        self.users = {}
        self.subjects = {}
        self.objects = {}
        self.config = config
        self.last_id = 0
        self.seen = set([])
        self.db_exists = False
        self.timeout = timeout  # wait x seconds to acquire db connection
        try:
            self.create_db()
            self.save()
        except sqlite3.OperationalError:
            self.db_exists = True

    def connect_db(self):
        return sqlite3.connect(self.config.db_path + self.config.db_name, timeout=self.timeout)

    def create_db(self):
        conn = self.connect_db()
        conn.execute('CREATE TABLE users (user_id PRIMARY KEY, user_score, ' + \
                     'confusion_matrix, history)')

        conn.execute('CREATE TABLE subjects (subject_id PRIMARY KEY, ' + \
                     'gold_label, score, retired, retired_as, seen, history)')

        conn.execute('CREATE TABLE thresholds (thresholds)')

        conn.execute('CREATE TABLE config (id PRIMARY KEY, user_default, ' + \
                     'workflow, p0, gamma, retirement_limit, db_path, ' + \
                     'db_name, timeout, last_id, seen)')

        conn.close()

    def load_users(self, users):
        for user in users:
            user_score = json.loads(user['user_score'])
            self.users[user['user_id']] = User(user_id=user['user_id'],
                                               classes=self.config.classes,
                                               gamma=self.config.gamma,
                                               user_default=user_score)
            self.users[user['user_id']].confusion_matrix = json.loads(user['confusion_matrix'])
            self.users[user['user_id']].history = json.loads(user['history'])

    def load_subjects(self, subjects):
        for subject in subjects:
            self.subjects[subject['subject_id']] = Subject(subject_id=subject['subject_id'],
                                                           classes=self.config.classes,
                                                           p0=self.config.p0)
            self.subjects[subject['subject_id']].score = subject['score']
            self.subjects[subject['subject_id']].gold_label = subject['gold_label']
            self.subjects[subject['subject_id']].retired = subject['retired']
            self.subjects[subject['subject_id']].retired_as = subject['retired_as']
            self.subjects[subject['subject_id']].seen = subject['seen']
            self.subjects[subject['subject_id']].history = json.loads(subject['history'])

    def load(self):
        def it(rows):
            for item in rows:
                yield dict(item)

        conn = self.connect_db()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute('SELECT * FROM config')
        config = dict(c.fetchone())

        swap = kSWAP(config=self.config,
                     timeout=config['timeout'])

        swap.last_id = config['last_id']
        swap.seen = set(json.loads(config['seen']))

        c.execute('SELECT * FROM users')
        swap.load_users(it(c.fetchall()))

        c.execute('SELECT * FROM subjects')
        swap.load_subjects(it(c.fetchall()))

        conn.close()

        return swap

    def dump_users(self):
        users = []
        for u in self.users.keys():
            users.append(self.users[u].dump())
        return users

    def dump_subjects(self):
        subjects = []
        for s in self.subjects.keys():
            subjects.append(self.subjects[s].dump())
        return subjects

    def dump_objects(self):
        objects = []
        for o in self.objects.keys():
            objects.append(self.objects[o].dump())
        return objects

    def dump_config(self):
        return (0, json.dumps(self.config.user_default), self.config.workflow,
                json.dumps(self.config.p0), self.config.gamma,
                self.config.retirement_limit, self.config.db_path,
                self.config.db_name, self.timeout, self.last_id,
                json.dumps(list(self.seen)))

    def save(self):
        conn = self.connect_db()
        c = conn.cursor()

        def zip_name(data):
            return [d.values() for d in data]

        c.executemany('INSERT OR REPLACE INTO users VALUES (?,?,?,?)',
                      self.dump_users())

        c.executemany('INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?)',
                      self.dump_subjects())

        c.execute('INSERT OR REPLACE INTO config VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                  self.dump_config())

        conn.commit()
        conn.close()

    def process_classification(self, cl, online=False):
        # check user is known
        try:
            self.users[cl.user_id]
        except KeyError:
            self.users[cl.user_id] = User(user_id=cl.user_id,
                                          classes=self.config.classes,
                                          gamma=self.config.gamma,
                                          user_default=self.config.user_default)
        # check subject is known
        try:
            self.subjects[cl.subject_id]
        except KeyError:
            self.subjects[cl.subject_id] = Subject(subject_id=cl.subject_id,
                                                   p0=self.config.p0,
                                                   classes=self.config.classes)

        self.subjects[cl.subject_id].update_score(cl.id, cl.label, self.users[cl.user_id])

        if self.subjects[cl.subject_id].gold_label in self.config.label_map.values() and online:
            gold_label = self.subjects[cl.subject_id].gold_label
            assert gold_label in self.config.label_map.values()
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
        self.users[cl.user_id].history.append((cl.subject_id, self.users[cl.user_id].user_score))
        self.last_id = cl.id
        self.seen.add(cl.id)

    def retire(self, subject_batch):
        to_retire = []
        for subject_id in subject_batch:
            try:
                subject = self.subjects[subject_id]
            except KeyError:
                print('Subject {} is missing.'.format(subject_id))
                continue
            # if subject.gold_label in (0, 1):
            #     # if this is a gold standard image never retire
            #     continue
            # if subject.score < self.config.thresholds[0]:
            #     subject.retired = True
            #     subject.retired_as = 0
            #     to_retire.append(subject_id)
            # elif subject.score > self.config.thresholds[1]:
            #     subject.retired = True
            #     subject.retired_as = 1
            #     to_retire.append(subject_id)

            if subject.gold_label in self.config.label_map.values():
                # if this is a gold standard image never retire
                continue
            subject_keys = json.loads(subject.score).keys()
            for key in subject_keys:
                if key == '0':
                    if (1 - subject.score[key]) < self.config.thresholds[0]:
                        subject.retired = True
                        subject.retired_as = key
                        to_retire.append(subject_id)
                else:
                    if subject.score[key] > self.config.thresholds[1]:
                        subject.retired = True
                        subject.retired_as = key
                        to_retire.append(subject_id)
        return to_retire

    def retire_classification_count(self, subject_batch):

        def majority_vote(sequence):
            occurence_count = Counter(sequence)
            return occurence_count.most_common(1)[0][0]

        to_retire = []
        for subject_id in subject_batch:
            try:
                subject = self.subjects[subject_id]
            except KeyError:
                print('Subject {} is missing.'.format(subject_id))
                continue
            if subject.seen >= self.config.retirement_limit:
                subject.retired = True
                subject.retired_as = majority_vote([h[3] for h in subject.history])
                to_retire.append(subject_id)
        return to_retire

    def process_classifications_from_csv_dump(self, path, online=False):
        with open(path, 'r') as csvdump:
            reader = csv.DictReader(csvdump)
            for row in reader:
                id = int(row['classification_id'])
                try:
                    assert int(row['workflow_id']) == self.config.workflow
                    # ignore repeat classifications of the same subject
                    if json.loads(row['metadata'])['seen_before']:
                        continue
                except KeyError as e:
                    print(e, row)
                    pass
                except AssertionError as e:
                    print(e, row)
                    continue
                try:
                    user_id = int(row['user_id'])
                except ValueError:
                    user_id = row['user_name']
                subject_id = int(row['subject_ids'])
                annotation = json.loads(row['annotations'])
                try:
                    cl = Classification(id,
                                        user_id,
                                        subject_id,
                                        annotation,
                                        label_map=self.config.label_map)
                except ValueError as e:
                    print(e)
                    continue
                self.process_classification(cl, online)
        try:
            self.retire(self.subjects.keys())
        except TypeError:
            self.retire_classification_count(self.subjects.keys())

    def get_golds(self, path):
        with open(path, 'r') as csvdump:
            reader = csv.DictReader(csvdump)
            for row in reader:
                subject_id = int(row['subject_id'])
                gold_label = int(row['gold'])
                self.subjects[subject_id] = Subject(subject_id,
                                                    classes=self.config.classes,
                                                    p0=self.config.p0,
                                                    gold_label=gold_label)

    def apply_golds(self, path):
        with open(path, 'r') as csvdump:
            reader = csv.DictReader(csvdump)
            for row in reader:
                id = int(row['classification_id'])
                try:
                    user_id = int(row['user_id'])
                except ValueError as e:
                    user_id = row['user_name']
                subject_id = int(row['subject_ids'])
                annotation = json.loads(row['annotations'])
                try:
                    assert int(row['workflow_id']) == self.config.workflow
                    # ignore repeat classifications of the same subject
                    if json.loads(row['metadata'])['seen_before']:
                        continue
                except KeyError as e:
                    print(e, row)
                    pass
                except AssertionError as e:
                    # print(e, row)
                    continue

                try:
                    cl = Classification(id,
                                        user_id,
                                        subject_id,
                                        annotation,
                                        label_map=self.config.label_map)
                except ValueError as e:
                    print(e)
                    continue
                try:
                    self.users[cl.user_id]
                except KeyError:
                    self.users[cl.user_id] = User(user_id=cl.user_id,
                                                  classes=self.config.classes,
                                                  gamma=self.config.gamma,
                                                  user_default=self.config.user_default)

                try:
                    gold_label = self.subjects[cl.subject_id].gold_label
                    assert gold_label in self.config.label_map.values()
                    self.users[cl.user_id].update_user_score(gold_label, cl.label)
                except AssertionError as e:
                    print(e)
                    continue
                except KeyError as e:
                    # print('420', e)
                    continue

    def run_offline(self, gold_csv, classification_csv):
        self.get_golds(gold_csv)
        self.apply_golds(classification_csv)
        self.process_classifications_from_csv_dump(classification_csv)

    def run_online(self, gold_csv, classification_csv):
        self.get_golds(gold_csv)
        self.process_classifications_from_csv_dump(classification_csv, online=True)
