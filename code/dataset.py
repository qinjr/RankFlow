import argparse
import os
import numpy as np
import yaml
import pandas as pd
import pickle as pkl
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

class IndDataset(Dataset):
    def __init__(self, config: dict, data_part: str, mode: str, random_neg_sample: bool,
                 generate = False) -> None:
        super().__init__()
        self.config = config
        self.data_part = data_part
        self.mode = mode

        self.uid = self.config['uid_col']
        self.iid = self.config['iid_col']
        self.ufeat = self.config['ufeat_cols']
        self.ifeat = self.config['ifeat_cols']
        self.label_col = self.config['label_col']

        if generate:
            print('dataset generating...')
            self._gen_user_set()
            self._gen_gt_dict()
            if mode == 'point':
                self._gen_point_wise_dataset(random_neg_sample)
            elif mode == 'point_pos':
                self._gen_pos_point_wise_dataset()
            elif mode == 'pair':
                self._gen_pair_wise_dataset(random_neg_sample)
            elif mode == 'list':
                self._gen_list_wise_dataset()
            
            if data_part == 'train':
                self._gen_pv_dict()
            print('dataset generated')
        
        print('dataset loading...')
        if self.data_part == 'train':
            file = '{}_set_{}_rns.pkl'.format(self.data_part, self.mode) if random_neg_sample else '{}_set_{}.pkl'.format(self.data_part, self.mode)
            with open(os.path.join(self.config['data_dir'], file), 'rb') as f:
                self.dataset_tuple = pkl.load(f)
        else:
            with open(os.path.join(self.config['data_dir'], '{}_set_{}.pkl'.format(self.data_part, self.mode)), 'rb') as f:
                self.dataset_tuple = pkl.load(f)
        
        self.dataset_tuple = [torch.from_numpy(t) for t in self.dataset_tuple]
        
        print('dataset loaded')

    # make it a pytorch dataset
    def __getitem__(self, idx):
        res = [t[idx] for t in self.dataset_tuple]
        return res  
    def __len__(self):
        return self.dataset_tuple[0].shape[0]

    def get_data(self) -> tuple:
        return self.dataset_tuple
    
    def _gen_point_wise_dataset(self, random_neg_sample = False) -> None:
        inter_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        # add inter_df with random neg sample if set
        if random_neg_sample:
            user_df = pd.read_csv(self.config['user_df'])
            item_df = pd.read_csv(self.config['item_df'])
            df = inter_df[[self.uid, self.iid, self.label_col]]
            df_pos = df.loc[df[self.label_col] == 1]
            print(len(df_pos))

            pos_dict = df_pos.groupby(self.uid)[self.iid].apply(list).to_dict()
            all_items = item_df[self.iid].unique()
            expand_df = []
            print('rs begin...')
            for uid in tqdm(pos_dict, total=len(pos_dict)):
                if uid in pos_dict:
                    pos_iids = pos_dict[uid]
                    for iid in pos_iids:
                        expand_df += [[uid, iid, 1]]
                        cnt = 0
                        while cnt != self.config['neg_sample']:
                            n = random.choice(all_items)
                            # if n not in pos_iids:
                            expand_df += [[uid, n, 0]]
                            cnt += 1
            print(len(expand_df))
            expand_df = pd.DataFrame(expand_df, columns=[self.uid, self.iid, self.label_col])
            inter_df = pd.merge(pd.merge(expand_df, user_df, on=self.uid), item_df, on=self.iid)

        # x = inter_df[[self.uid] + self.ufeat + [self.iid] + self.ifeat].to_numpy(dtype=int)
        y = inter_df[self.label_col].to_numpy(dtype=int)

        # for those dual-tower models like DSSM
        x_user = inter_df[[self.uid] + self.ufeat].to_numpy(dtype=int)
        x_item = inter_df[[self.iid] + self.ifeat].to_numpy(dtype=int)

        if random_neg_sample:
            with open(os.path.join(self.config['data_dir'], '{}_set_point_rns.pkl'.format(self.data_part)), 'wb') as f:
                pkl.dump([x_user, x_item, y], f)
        else:    
            with open(os.path.join(self.config['data_dir'], '{}_set_point.pkl'.format(self.data_part)), 'wb') as f:
                pkl.dump([x_user, x_item, y], f)
        print('point-wise {} set generated'.format(self.data_part))

    def _gen_pos_point_wise_dataset(self) -> None:
        inter_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        inter_df_pos = inter_df.loc[inter_df[self.label_col] == 1]

        x_user = inter_df_pos[[self.uid] + self.ufeat].to_numpy(dtype=int)
        x_item = inter_df_pos[[self.iid] + self.ifeat].to_numpy(dtype=int)

        with open(os.path.join(self.config['data_dir'], '{}_set_point_pos.pkl'.format(self.data_part)), 'wb') as f:
            pkl.dump([x_user, x_item], f)
        print('positive point-wise {} set generated'.format(self.data_part))


    def _gen_pair_wise_dataset(self, random_neg_sample = False) -> None:
        inter_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        user_df = pd.read_csv(self.config['user_df'])
        item_df = pd.read_csv(self.config['item_df'])
        df = inter_df[[self.uid, self.iid, self.label_col]]
        df_pos = df.loc[df[self.label_col] == 1]
        df_neg = df.loc[df[self.label_col] == 0]

        pos_dict = df_pos.groupby(self.uid)[self.iid].apply(list).to_dict()
        neg_dict = df_neg.groupby(self.uid)[self.iid].apply(list).to_dict()
        all_items = item_df[self.iid].unique()

        pair_wise_df = []
        for uid in tqdm(pos_dict, total=len(pos_dict)):
            if uid in pos_dict and uid in neg_dict:
                pos_iids = list(set(pos_dict[uid]))
                neg_iids = list(set(neg_dict[uid]))
                
                for iid in pos_iids:
                    if random_neg_sample:
                        cnt = 0
                        while cnt != self.config['neg_sample']:
                            n = random.choice(neg_dict[uid])
                            if n not in pos_iids:
                                pair_wise_df += [[uid, iid, n]]
                                cnt += 1
                    else:
                        pair_wise_df += [[uid, iid, n] for n in neg_iids]
        
        pair_wise_df = pd.DataFrame(pair_wise_df, columns=[self.uid, self.iid, self.iid + '_neg'])

        x_user = pd.merge(pair_wise_df[[self.uid]], user_df, on=self.uid).to_numpy()
        x_item = pd.merge(pair_wise_df[[self.iid]], item_df, on=self.iid).to_numpy()
        x_item_neg = pd.merge(pair_wise_df[[self.iid + '_neg']], item_df, left_on=self.iid + '_neg', right_on=self.iid)[[self.iid] + self.ifeat].to_numpy()

        if random_neg_sample:
            with open(os.path.join(self.config['data_dir'], '{}_set_pair_rns.pkl'.format(self.data_part)), 'wb') as f:
                pkl.dump([x_user, x_item, x_item_neg], f)
        else:
            with open(os.path.join(self.config['data_dir'], '{}_set_pair.pkl'.format(self.data_part)), 'wb') as f:
                pkl.dump([x_user, x_item, x_item_neg], f)
        print('pair-wise {} set generated'.format(self.data_part))


    '''only for D testing'''
    def _gen_list_wise_dataset(self) -> None:
        inter_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        user_df = pd.read_csv(self.config['user_df'])
        item_df = pd.read_csv(self.config['item_df'])
        df = inter_df[[self.uid, self.iid, self.label_col]]
        df_pos = df.loc[df[self.label_col] == 1]
        df_neg = df.loc[df[self.label_col] == 0]

        pos_dict = df_pos.groupby(self.uid)[self.iid].apply(list).to_dict()
        neg_dict = df_neg.groupby(self.uid)[self.iid].apply(list).to_dict()
        all_items = item_df[self.iid].unique()

        uids = []
        iids = []
        labels = []
        for uid in tqdm(pos_dict, total=len(pos_dict)):
            pos_list = pos_dict[uid]
            if uid in neg_dict:
                neg_list = neg_dict[uid]
            else:
                neg_list = []
            
            pos_label = [1] * len(pos_list)
            neg_label = [0] * len(neg_list)
            ulist = pos_list + neg_list
            ulabel = pos_label + neg_label
            list_len = self.config['test_list_len']
            while len(ulist) < list_len:
                neg_item = random.choice(all_items)
                if neg_item not in ulist:
                    ulist += [neg_item]
                    ulabel += [0]
            ulist = ulist[:list_len]
            ulabel = ulabel[:list_len]
            uids += [uid] * list_len
            iids += ulist
            labels += ulabel
            
        new_df = pd.DataFrame(zip(uids, iids, labels), columns=[self.uid, self.iid, self.label_col])
        inter_df = pd.merge(pd.merge(new_df, item_df, on=self.iid), user_df, on=self.uid)
        inter_df = inter_df.sort_values(by=[self.uid, self.label_col], ascending=False)
        
        # x = inter_df[[self.uid] + self.ufeat + [self.iid] + self.ifeat].to_numpy(dtype=int)
        y = inter_df[self.label_col].to_numpy(dtype=int)
        x_user = inter_df[[self.uid] + self.ufeat].to_numpy(dtype=int)
        x_item = inter_df[[self.iid] + self.ifeat].to_numpy(dtype=int)

        with open(os.path.join(self.config['data_dir'], '{}_set_list.pkl'.format(self.data_part)), 'wb') as f:
            pkl.dump([x_user, x_item, y], f)
        print('list-wise {} set generated'.format(self.data_part))

    '''only for train set'''
    def _gen_pv_dict(self) -> None:
        train_set = pd.read_csv(self.config['train_inter'])
        train_set = train_set.groupby(self.uid)[self.iid].apply(list)
        pv_dict = train_set.to_dict()

        with open(os.path.join(self.config['data_dir'], 'pv_dict.pkl'), 'wb') as f:
            pkl.dump(pv_dict, f)
        print('pv dict is generated')

    def _gen_user_set(self) -> None:
        log_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        log_df = log_df.loc[log_df[self.label_col] == 1]
        user_df = log_df.drop_duplicates(self.uid)[[self.uid] + self.ufeat]
        user_set = user_df.to_numpy(dtype=int)

        with open(os.path.join(self.config['data_dir'], '{}_user_set.pkl'.format(self.data_part)), 'wb') as f:
            pkl.dump(user_set, f)
        print('{} user set is generated'.format(self.data_part))
    
    # gt is for 'ground truth'
    def _gen_gt_dict(self) -> None:
        log_df = pd.read_csv(self.config['{}_inter'.format(self.data_part)])
        gt_dict = dict()
        log_df = log_df.loc[log_df[self.label_col] == 1]
        log_df = log_df.groupby(self.uid)[self.iid].apply(list)
        gt_dict = log_df.to_dict()
        
        with open(os.path.join(self.config['data_dir'], '{}_gt.pkl'.format(self.data_part)), 'wb') as f:
            pkl.dump(gt_dict, f)
        print('ground truth of {} is generated'.format(self.data_part))

class UserSet(object):
    def __init__(self, config: dict, data_part: str) -> None:
        super().__init__()
        self.config = config
        self.data_part = data_part

        # read user set file
        with open(os.path.join(self.config['data_dir'], '{}_user_set.pkl'.format(data_part)), 'rb') as f:
            self.user_set = pkl.load(f)
    
    def get_user_set(self):
        return [self.user_set]

class ItemSet(object):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        
        # read all items
        item_df = pd.read_csv(self.config['item_df'])    
        self.item_set = item_df.to_numpy(dtype=int)
    
    def get_item_set(self):
        return [self.item_set]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ml-1m')
    args = parser.parse_args()

    # go to root path
    root_path = '..'
    os.chdir(root_path)

    config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = IndDataset(config, 'train', 'point_pos', False, True)
    t = dataset.get_data()
    for i in t:
        if i is not None:
            print(i.shape)
        else:
            print('no user hist')
    print('---------------------------')

    dataset = IndDataset(config, 'train', 'point', False, True)
    t = dataset.get_data()
    for i in t:
        if i is not None:
            print(i.shape)
        else:
            print('no user hist')
    print('---------------------------')

    # dataset = IndDataset(config, 'train', 'pair', False, True)
    # t = dataset.get_data()
    # for i in t:
    #     if i is not None:
    #         print(i.shape)
    #     else:
    #         print('no user hist')
    # print('---------------------------')

    # # dataset = IndDataset(config, 'train', 'pair', True, True)
    # # t = dataset.get_data()
    # # for i in t:
    # #     if i is not None:
    # #         print(i.shape)
    # #     else:
    # #         print('no user hist')
    # # print('---------------------------')


    dataset = IndDataset(config, 'test', 'point', False, True)
    # dataset = IndDataset(config, 'unpv', 'point', False, True)
    dataset = IndDataset(config, 'test', 'list', False, True)
    # dataset = IndDataset(config, 'unpv', 'list', False, True)


    # train_dataset = IndDataset(config, 'train', 'point', False)
    # data_tuple = train_dataset.get_data()
    # x_user, x_item, y = data_tuple
    # print(x_user.shape)
    # print(x_item.shape)
    # print(y.shape)

    # for i in range(config['user_num_fields']):
    #     print('min value of feature {} is {}'.format(i, torch.min(x_user[:,i])))
    #     print(torch.max(x_user[:,i]))
    #     print(torch.unique(x_user[:,i]).shape)
    # for i in range(config['item_num_fields']):
    #     print(torch.min(x_item[:,i]))
    #     print(torch.max(x_item[:,i]))
    #     print(torch.unique(x_item[:,i]).shape)

    # print(y[y==0].shape)
    # print(y[y==1].shape)
