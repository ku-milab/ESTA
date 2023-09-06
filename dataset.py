import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, maskers, datasets
from sklearn.model_selection import StratifiedKFold









class DatasetHCPTask(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, standardized_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()

        self.filename = 'train_hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi == 'schaefer':
            self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'aal':
            self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'destrieux':
            self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'harvard_oxford':
            self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm',
                                                           data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.standardized_length = standardized_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))


        else:
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'TASK', task)) if f.endswith('nii.gz')]
                img_list.sort()
                for subject in tqdm(img_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    timeseries = roi_masker.fit_transform(image.load_img(os.path.join(self.sourcedir, 'img', 'TASK', task, subject)))
                    if not len(timeseries) == task_timepoints[task]:
                        print(f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(timeseries)
                    self.label_list.append(task)
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, f'{self.filename}.pth'))




        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None




    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)




    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list( self.k_fold.split(self.timeseries_list, self.label_list) )[fold]


        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False




    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.standardized_length is None:
            timeseries = timeseries[:self.standardized_length]
        task = self.label_list[self.fold_idx[idx]]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}











# we processed the test data using stratified sampling
class DatasetHCPTask_test(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, standardized_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()

        self.filename = 'test_hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi == 'schaefer':
            self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'aal':
            self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'destrieux':
            self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'harvard_oxford':
            self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.standardized_length = standardized_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)


        self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))




        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None




    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)





    def __getitem__(self, idx):
        timeseries = self.timeseries_list[idx]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.standardized_length is None:
            timeseries = timeseries[:self.standardized_length]
        task = self.label_list[idx]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}










