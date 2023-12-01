import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from data_preprocessing import DrugDataset, DrugDataLoader

class Planetoid(object):
    def __init__(self, data_name, fold='fold0', batch_size=512, save_suffix=''):
        self.data_name = data_name
        self.data_logger_save = data_name+save_suffix
        self.batch_size = batch_size

        if data_name.find('drugbank') > -1:
            train_data_loader,val_data_loader,test_data_loader = self.get_data(fold)
        elif data_name.find('twosides') > -1:
            train_data_loader, val_data_loader, test_data_loader = self.get_data_twosides(fold)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.num_features = self.train_data_loader.dataset.n_atom_feats
        self.rel_total = self.train_data_loader.dataset.rel_total
        self.num_labels = 1

    def get_data(self, fold):
        transductive_flag = os.path.exists(os.path.join('dataset', self.data_name, fold, 'test.csv'))
        self.transductive_flag = transductive_flag
        if transductive_flag == True:
            #transductive
            df_ddi_train = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'train.csv'))
            df_ddi_test = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'test.csv'))

            train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
            train_tup, val_tup = self.split_train_valid(train_tup, 2, val_ratio=0.2)
            test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

            train_data = DrugDataset(train_tup)
            val_data = DrugDataset(val_tup, disjoint_split=False)
            test_data = DrugDataset(test_tup, disjoint_split=False)

            print(
                f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

            train_data_loader = DrugDataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_data_loader = DrugDataLoader(val_data, batch_size=self.batch_size * 3, num_workers=2)
            test_data_loader = DrugDataLoader(test_data, batch_size=self.batch_size * 3, num_workers=2)

            return train_data_loader,val_data_loader,test_data_loader
        else:
            # inductive
            df_ddi_train = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'train.csv'))
            df_ddi_s1 = pd.read_csv(os.path.join('dataset', self.data_name, fold, 's1.csv'))
            df_ddi_s2 = pd.read_csv(os.path.join('dataset', self.data_name, fold, 's2.csv'))

            train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
            s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
            s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

            train_data = DrugDataset(train_tup)
            s1_data = DrugDataset(s1_tup, disjoint_split=True)
            s2_data = DrugDataset(s2_tup, disjoint_split=True)

            print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

            train_data_loader = DrugDataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            s1_data_loader = DrugDataLoader(s1_data, batch_size=self.batch_size * 3, num_workers=2)
            s2_data_loader = DrugDataLoader(s2_data, batch_size=self.batch_size * 3, num_workers=2)

            return train_data_loader,s1_data_loader,s2_data_loader

    def split_train_valid(self, data, fold, val_ratio=0.2):
        data = np.array(data)
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
        train_tup = data[train_index]
        val_tup = data[val_index]
        train_tup = [(tup[0],tup[1],int(tup[2]))for tup in train_tup ]
        val_tup = [(tup[0],tup[1],int(tup[2]))for tup in val_tup ]

        return train_tup, val_tup

    ### same as the previous methods, like DSN-DDI
    def get_data_twosides(self, fold):
        transductive_flag = os.path.exists(os.path.join('dataset', self.data_name, fold, 'test.csv'))
        self.transductive_flag = transductive_flag
        if transductive_flag == True:
            #transductive
            df_ddi_train = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'train.csv'))
            df_ddi_test = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'test.csv'))

            train_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'], df_ddi_train['Neg samples'])]
            train_tup, val_tup = self.split_train_valid_twosides(train_tup, 2, val_ratio=0.2)
            test_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'], df_ddi_train['Neg samples'])]

            train_data = DrugDataset(train_tup)
            val_data = DrugDataset(val_tup, disjoint_split=False)
            test_data = DrugDataset(test_tup, disjoint_split=False)

            print(
                f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

            train_data_loader = DrugDataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_data_loader = DrugDataLoader(val_data, batch_size=self.batch_size * 3, num_workers=2)
            test_data_loader = DrugDataLoader(test_data, batch_size=self.batch_size * 3, num_workers=2)

            return train_data_loader,val_data_loader,test_data_loader
        else:
            # inductive
            df_ddi_train = pd.read_csv(os.path.join('dataset', self.data_name, fold, 'train.csv'))
            df_ddi_s1 = pd.read_csv(os.path.join('dataset', self.data_name, fold, 's1.csv'))
            df_ddi_s2 = pd.read_csv(os.path.join('dataset', self.data_name, fold, 's2.csv'))

            train_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'], df_ddi_train['Neg samples'])]
            s1_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'], df_ddi_train['Neg samples'])]
            s2_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'], df_ddi_train['Neg samples'])]

            train_data = DrugDataset(train_tup)
            s1_data = DrugDataset(s1_tup, disjoint_split=True)
            s2_data = DrugDataset(s2_tup, disjoint_split=True)

            print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

            train_data_loader = DrugDataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            s1_data_loader = DrugDataLoader(s1_data, batch_size=self.batch_size * 3, num_workers=2)
            s2_data_loader = DrugDataLoader(s2_data, batch_size=self.batch_size * 3, num_workers=2)

            return train_data_loader,s1_data_loader,s2_data_loader

    def split_train_valid_twosides(self, data, fold, val_ratio=0.2):
        data = np.array(data)
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
        train_tup = data[train_index]
        val_tup = data[val_index]
        train_tup = [(tup[0], tup[1], int(tup[2]),tup[3]) for tup in train_tup]
        val_tup = [(tup[0], tup[1], int(tup[2]),tup[3]) for tup in val_tup]

        return train_tup, val_tup