class Config(object):
    def __init__(self, workflow=16311):
        self.project = 11726
        self.workflow = workflow
        self.swap_path = './'
        self.data_path = './csv/'
        self.db_name = 'offline_swap.db'
        self.db_path = './db/'

        self.classes      = ['0', '1']
        self.label_map    = {'0': 0, '1': 1}
        self.user_default = {'0': [0.50, 0.50], '1': [0.50, 0.50]}
        self.p0           = {'0': 0.9, '1': 0.1}
        self.thresholds   = (0.01, 0.9)
        self.retirement_limit = 10
        self.gamma = 1
