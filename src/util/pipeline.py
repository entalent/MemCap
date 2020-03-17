import os
import shutil
import sys
import argparse
import datetime
import json
import zipfile
from abc import abstractmethod

import torch
from tensorboardX import SummaryWriter


class BasePipeline:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        self.args = parser.parse_args()

        run_name = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + (
            '' if len(self.args.run_name) == 0 else '_' + self.args.run_name)

        save_folder = os.path.join('..', 'save', run_name)

        self.run_name = run_name
        self.save_folder = save_folder

        self.writer = SummaryWriter(os.path.join(save_folder, 'log'))

    def add_arguments(self, parser):
        parser.add_argument('-run_name', default='', type=str)

    @staticmethod
    def zip_source_code(folder, target_zip_file):
        f_zip = zipfile.ZipFile(target_zip_file, 'w', zipfile.ZIP_STORED)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.py'):
                    f_zip.write(os.path.join(root, file))
        f_zip.close()

    def run(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        print('run:', self.run_name)
        print('save folder:', self.save_folder)
        print('args:', json.dumps(self.args.__dict__, indent=4))

        with open(os.path.join(self.save_folder, 'args'), 'w') as f:
            f.write('cwd: ' + os.getcwd() + '\n')
            f.write('cmd: ' + ' '.join(sys.argv) + '\n')
            f.write('args: ' + str(self.args) + '\n')
        self.zip_source_code(folder='.', target_zip_file=os.path.join(self.save_folder, 'src.zip'))


    def save_model(self, save_path, state_dict):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(state_dict, save_path)
        print('model saved at {}'.format(save_path))

    def load_model(self, save_path):
        state_dict = torch.load(save_path)
        print('loaded model at {}'.format(save_path))
        return state_dict


class SupervisedPipeline(BasePipeline):
    """
    a general training / testing pipeline for supervised learning
    """
    def __init__(self):
        super().__init__()
        self.checkpoint_folder = os.path.join(self.save_folder, 'checkpoint')

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('action', default='', type=str, help='action to execute, i.e. train, test')
        parser.add_argument('-checkpoint', default='', type=str, help='path to saved model')
        parser.add_argument('-max_epoch', default=1, type=int, help='maximum epochs to run (start from 1)')
        parser.add_argument('-save_every_epoch', default=0, type=int, help='if 1, save after every epoch')

    def run(self):
        super().run()
        action = self.args.action
        kwargs = {}
        if action == 'test':
            kwargs['should_init'] = True
        self.__getattribute__(action)(**kwargs)

    @abstractmethod
    def init_data(self):
        pass

    @abstractmethod
    def init_model(self, state_dict=None):
        """
        initialize or restore model, optimizer, loss function
        :param state_dict:
        :return:
        """
        pass

    def _init_model(self):
        """
        sub class should not override this method
        :return:
        """
        if len(self.args.checkpoint) > 0:
            state_dict = self.load_model(self.args.checkpoint)
        else:
            state_dict = None
        self.init_model(state_dict)

    def train(self):
        self.global_step = 0
        self.epoch = 1

        self.init_data()
        self._init_model()

        while self.epoch <= self.args.max_epoch:
            print('epoch: {}'.format(self.epoch))
            self.train_epoch()
            self.test(should_save=True)
            self.epoch += 1

    def test(self, should_init=False, should_save=False):
        if should_init:
            self.init_data()
            self._init_model()
        is_best = self.test_epoch()
        if should_save:
            if self.args.save_every_epoch:
                model_path = os.path.join(self.checkpoint_folder, 'checkpoint_epoch_{}'.format(self.epoch))
                self.save_model(model_path, self.get_state_dict())
            else:
                model_path = os.path.join(self.checkpoint_folder, 'checkpoint_latest')
                self.save_model(model_path, self.get_state_dict())
                if is_best:
                    shutil.copy(src=model_path, dst=os.path.join(self.checkpoint_folder, 'checkpoint_best'))

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def test_epoch(self):
        """
        :return: whether this test is the best result
        """
        pass

    @abstractmethod
    def get_state_dict(self):
        pass

    def save_model(self, save_path, state_dict):
        if state_dict is None:
            print('state_dict not saved')
            return
        state_dict['global_step'] = self.global_step
        state_dict['epoch'] = self.epoch
        super().save_model(save_path, state_dict)

    def load_model(self, save_path):
        state_dict = super().load_model(save_path)
        self.epoch = state_dict['epoch'] + 1
        self.global_step = state_dict['global_step']
        return super().load_model(save_path)

