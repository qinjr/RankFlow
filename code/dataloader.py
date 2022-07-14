import numpy as np
import argparse
import os
from tqdm.std import tqdm
import yaml
import time
from dataset import IndDataset, UserSet, ItemSet

class Dataloader(object):
    def __init__(self, dataset_tuple, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_tuple = dataset_tuple
        
        
        self.dataset_size = len(self.dataset_tuple[0])
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        self.refresh()
    
    def __len__(self):
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.dataset_size)
        for i in range(len(self.dataset_tuple)):
            self.dataset_tuple[i] = self.dataset_tuple[i][perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1
        batch_data = []
        for i in range(len(self.dataset_tuple)):
            batch_data.append(self.dataset_tuple[i][left:right])
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ml-1m')
    args = parser.parse_args()

    # go to root path
    root_path = '..'
    os.chdir(root_path)

    config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = IndDataset(config, 'train', 'point', True)
    dataset_tuple = dataset.get_data()

    dl = IndDataset(dataset_tuple, 256, True)

    t = time.time()
    for i in range(2):
        for batch_data in tqdm(dl, total=dl.total_step):
            continue
        print('epoch %d time is %.2f seconds:' % (i, time.time() - t))
        dl.refresh()
        t = time.time()

    for batch_data in dl:
        for d in batch_data:
            print(d.shape)
        break

    print('testing user and item set loader')
    user_set = UserSet(config, 'train').get_user_set()
    dl = Dataloader(user_set, 256, False)
    for users in dl:
        print(users[0].shape)
    
    item_set = ItemSet(config).get_item_set()
    dl = Dataloader(item_set, 256, False)
    for items in dl:
        print(items[0].shape)
    

