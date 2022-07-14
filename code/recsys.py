import pickle as pkl
from numpy.core.fromnumeric import sort
import torch
import numpy as np
from numpy.random import RandomState
import argparse
import os
import time
import datetime
import yaml
import faiss
from logging import getLogger

from tqdm.std import tqdm
from recbole.utils import set_color, init_seed
from utils.log import dict2str, combo_dict
from warmup import get_model
from utils.yaml_loader import get_yaml_loader
from dataset import UserSet, ItemSet
from dataloader import Dataloader
from utils.evaluations import TopKMetric, PointMetric

class RecSys(object):
    def __init__(self, model_configs, dataset_config, joint_train_config, save_model_dir = './saved_models/ind', load_models = True):
        self.model_configs = model_configs
        self.dataset_config = dataset_config
        self.joint_train_config = joint_train_config
        self.save_model_dir = save_model_dir

        self.item_set = ItemSet(dataset_config).get_item_set()

        self.batch_size = self.joint_train_config['eval_batch_size']
        self.rec_sizes = self.joint_train_config['rec_sizes']
        self.score_sizes = self.joint_train_config['score_sizes']
        self.topk_dict = self.joint_train_config['topk_dict']
        self.stage_names = self.joint_train_config['stage_names']
        self.eval_modes = self.joint_train_config['eval_modes']
        self.involve_bid = self.joint_train_config['involve_bid']
        self.logger = getLogger()
        self.device = torch.device(joint_train_config['device'])
        self.load_models = load_models

        if self.joint_train_config['have_hist']:
            with open(self.joint_train_config['hist_dict'] + 'train.pkl', 'rb') as f:
                self.hist_dict_train = pkl.load(f)
            with open(self.joint_train_config['hist_dict'] + 'test.pkl', 'rb') as f:
                self.hist_dict_test = pkl.load(f)

            with open(self.joint_train_config['hist_len_dict'] + 'train.pkl', 'rb') as f:
                self.hist_len_dict_train = pkl.load(f)
            with open(self.joint_train_config['hist_len_dict'] + 'test.pkl', 'rb') as f:
                self.hist_len_dict_test = pkl.load(f)

        self.loaded_models = []
        self.model_names = []
        for model_config in model_configs:
            self.loaded_models.append(self._resume_checkpoint(model_config, dataset_config))
            self.model_names.append(model_config['model_name'])
        self.recall_model = self.loaded_models[0]
        print('Multi-stage models have all been loaded')
        self._build_faiss_index()


    def _resume_checkpoint(self, model_config, dataset_config, resume_file = None):
        if resume_file is not None:
            checkpoint = torch.load(resume_file, map_location=self.device)
        else:
            ckpt_dir = self.save_model_dir
            files = []
            times = []
            for file in os.listdir(ckpt_dir):
                if file.startswith(model_config['model_name'].lower() + '-' + dataset_config['dataset_name']):
                    files.append(file)
                    time_str = '-'.join(file.split('-')[-5:])[:-4]
                    times.append(time.mktime(datetime.datetime.strptime(time_str, '%b-%d-%Y_%H-%M-%S').timetuple()))
            f = os.path.join(ckpt_dir, files[np.argmax(times)])
            print('loaded path: {}'.format(f))
            checkpoint = torch.load(f, map_location=self.device)

        model = get_model(model_config['model_name'], model_config, self.dataset_config).to(self.device)
        # load architecture params from checkpoint
        if self.load_models:
            model.load_state_dict(checkpoint['state_dict'])
            print('Model: {} has been loaded'.format(model_config['model_name']))
        return model

    def _build_faiss_index(self):
        self.recall_model.eval()
        item_repres = []
        item_loader = Dataloader(self.item_set, self.batch_size, False)
        print('building faiss index...')
        for batch_idx, batch_data in enumerate(tqdm(item_loader, total=item_loader.total_step, ncols=100, position=0, leave=True)):
            x_item = torch.from_numpy(batch_data[0]).to(self.device)
            item_repre = self.recall_model.get_item_repre(x_item)
            item_repres.append(item_repre)
        self.item_repres = torch.cat(item_repres).cpu().detach().numpy()

        self.index = faiss.IndexFlatIP(item_repre.shape[1])
        # faiss.normalize_L2(self.item_repres)
        self.index.add(self.item_repres)
        print('index ntotal is: {}'.format(self.index.ntotal))
        # torch.cuda.empty_cache()

    def _gen_recall_res(self, data_part):
        self.recall_model.eval()
        user_repres = []
        user_set = UserSet(self.dataset_config, data_part).get_user_set()
        user_loader = Dataloader(user_set, self.batch_size, False)
        print('generate recall results...')
        for batch_idx, batch_data in enumerate(tqdm(user_loader, total=user_loader.total_step, ncols=100, position=0, leave=True)):
            x_user = torch.from_numpy(batch_data[0]).to(self.device)
            user_hist = None
            hist_len = None
            if self.recall_model.use_hist:
                user_hist, hist_len = self._get_user_hist(x_user, data_part)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)
            user_repre = self.recall_model.get_user_repre(x_user, user_hist, hist_len)
            user_repres.append(user_repre)
        self.user_repres = torch.cat(user_repres).cpu().detach().numpy()

        recall_size = self.score_sizes['recall']
        print('faiss searching begin...')
        # faiss.normalize_L2(self.user_repres)
        recall_score, i  = self.index.search(self.user_repres, k=recall_size)
        print('faiss searching finished')
        i = i.flatten().reshape([-1,1])
        recalled_items = np.take_along_axis(self.item_set[0], i, axis=0).reshape([-1, recall_size, self.dataset_config['item_num_fields']])
        recalled_items_iid = np.expand_dims(recalled_items[:,:,0], 2)

        users_tile = user_set[0].reshape([-1, 1, user_set[0].shape[1]])
        users_tile = np.tile(users_tile, (1, recall_size, 1))
        users_tile_iid = np.expand_dims(users_tile[:,:,0], axis=2)

        recalled_ui_pair = np.concatenate((users_tile_iid, recalled_items_iid), axis=2)
        return users_tile, recalled_items, recalled_ui_pair, recall_score.reshape([-1,])

    def _get_labels(self, recalled_ui_pair, data_part):
        with open(os.path.join(self.dataset_config['data_dir'], '{}_gt.pkl'.format(data_part)), 'rb') as f:
            gt_dict = pkl.load(f)

        labels = np.zeros(recalled_ui_pair.shape[:2], dtype=np.float32)
        total_user_num = recalled_ui_pair.shape[0]
        recall_size = self.score_sizes['recall']
        total_relevant_num = np.ones((total_user_num,))

        print('get labels for all recalled items...')
        cnt = 0
        for i in range(total_user_num):
            total_relevant_num[i] = len(gt_dict[recalled_ui_pair[i][0][0]])
            for j in range(recall_size):
                if recalled_ui_pair[i][j][1] in gt_dict[recalled_ui_pair[i][j][0]]:
                    cnt += 1
                    labels[i][j] = 1
        print('total pos number: {}'.format(cnt))
        return labels, total_relevant_num

    def _get_stage_scoring_data(self, stage_idx, users_tile, candidate_items, data_part):
        self.loaded_models[stage_idx].eval()
        # reshape the inputs
        users_tile = users_tile.reshape([-1, self.dataset_config['user_num_fields']])
        candidate_items = candidate_items.reshape([-1, self.dataset_config['item_num_fields']])

        preds = []
        dataset_tuple = [users_tile, candidate_items, np.concatenate((users_tile, candidate_items), axis=1)]
        dl = Dataloader(dataset_tuple, self.batch_size, False)
        print('begin forward calculation of stage {}...'.format(self.stage_names[stage_idx]))
        for batch_data in tqdm(dl, total=dl.total_step, ncols=100, position=0, leave=True):
            x_user, x_item, x = batch_data
            user_hist = None
            hist_len = None
            if self.loaded_models[stage_idx].use_hist:
                user_hist, hist_len = self._get_user_hist(x_user, data_part)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)
            x_user = torch.from_numpy(x_user).to(self.device)
            x_item = torch.from_numpy(x_item).to(self.device)
            
            pred = self.loaded_models[stage_idx](x_user, x_item, user_hist, hist_len)

            pred = pred.cpu().detach().numpy()
            preds.append(pred)#pred if preds is None else np.concatenate((preds, pred), axis=0)
    
        return np.concatenate(preds, axis=0)

    def _test_one_stage(self, preds, bids, labels, total_relevant_num, stage_name):
        metrics = TopKMetric(self.topk_dict[stage_name], self.rec_sizes[stage_name], \
                        labels, preds, bids, total_relevant_num, stage_name) if self.eval_modes[stage_name] == 'list' else PointMetric(labels, preds)
        eval_result = metrics.get_metrics()
        return eval_result
    
    # for testing
    def test_all_stages(self, data_part):
        users_tile, candidate_items, recalled_ui_pair, recall_score = self._gen_recall_res(data_part)
        labels, total_relevant_num = self._get_labels(recalled_ui_pair, data_part)
        recsys_evals = {}

        for i, stage_name in enumerate(self.stage_names):
            # model = self.loaded_models[i]
            if stage_name == 'recall':
                preds = recall_score
            else:
                preds = self._get_stage_scoring_data(i, users_tile, candidate_items, data_part)
            
            # get this stage results
            preds = preds.reshape([-1, self.score_sizes[stage_name]])
            if stage_name == 'ranking' and self.involve_bid:
                random_state = np.random.RandomState(self.joint_train_config['seed'])
                bids = random_state.lognormal(3, 0.5, preds.shape)
                preds = preds * bids
                # print(bids)
            else:
                bids = None
            sorted_index = np.argsort(-preds, axis=1)[:,:self.rec_sizes[self.stage_names[i]]]
            select_preds = np.take_along_axis(preds, sorted_index, axis=1).reshape([-1,])
            select_labels = np.take_along_axis(labels, sorted_index, axis=1).reshape([-1,])
            eval_result = self._test_one_stage(select_preds, bids, select_labels, total_relevant_num, stage_name)
            recsys_evals[stage_name] = eval_result
            
            # input of next stage
            sorted_index_user = np.tile(np.expand_dims(sorted_index,axis=2),(1,1,self.dataset_config['user_num_fields']))
            sorted_index_item = np.tile(np.expand_dims(sorted_index,axis=2),(1,1,self.dataset_config['item_num_fields']))
            
            users_tile = np.take_along_axis(users_tile, sorted_index_user, axis=1)
            candidate_items = np.take_along_axis(candidate_items, sorted_index_item, axis=1)
            labels = np.take_along_axis(labels, sorted_index, axis=1)
            
        return recsys_evals

    # for training
    def gen_stage_data(self, data_part):
        stage_data = []
        users_tile, candidate_items, recalled_ui_pair, recall_score = self._gen_recall_res(data_part)
        labels, _ = self._get_labels(recalled_ui_pair, data_part)
        
        for i, stage_name in enumerate(self.stage_names):
            # r stands for reshape
            users_tile_r = users_tile.reshape([-1, self.dataset_config['user_num_fields']])
            candidate_items_r = candidate_items.reshape([-1, self.dataset_config['item_num_fields']])
            labels_r = labels.reshape([-1,])

            stage_data.append([users_tile_r, candidate_items_r, labels_r])

            # get next stage data
            if stage_name == 'recall':
                preds = recall_score
            else:
                preds = self._get_stage_scoring_data(i, users_tile, candidate_items, data_part)
            preds = preds.reshape([-1, self.score_sizes[stage_name]])

            sorted_index = np.argsort(-preds, axis=1)[:,:self.rec_sizes[self.stage_names[i]]]
            sorted_index_user = np.tile(np.expand_dims(sorted_index,axis=2),(1,1,self.dataset_config['user_num_fields']))
            sorted_index_item = np.tile(np.expand_dims(sorted_index,axis=2),(1,1,self.dataset_config['item_num_fields']))
            
            users_tile = np.take_along_axis(users_tile, sorted_index_user, axis=1)
            candidate_items = np.take_along_axis(candidate_items, sorted_index_item, axis=1)
            labels = np.take_along_axis(labels, sorted_index, axis=1)
        
        return stage_data

    def update_model(self, stage_idx, new_model):
        self.loaded_models[stage_idx] = new_model
    
    def update_faiss_index(self):
        self._build_faiss_index()

    def get_model_names(self):
        return self.model_names
    
    def get_model(self, idx):
        return self.loaded_models[idx]
    
    def _get_user_hist(self, x_user, stage = 'train'):
        uids = x_user[:,0].tolist()
        user_history = []
        hist_len = []
        if stage == 'train':
            for uid in uids:
                user_history.append(self.hist_dict_train[uid])
                hist_len.append(self.hist_len_dict_train[uid])
        elif stage == 'test':
            for uid in uids:
                user_history.append(self.hist_dict_test[uid])
                hist_len.append(self.hist_len_dict_test[uid])
        return torch.tensor(user_history), torch.tensor(hist_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ms', '--models', type=str, help='models name', default='dssm,fm,deepfm')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ml-1m')
    args = parser.parse_args()

    # go to root path
    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    joint_train_config_path = os.path.join('configs/joint_train_configs', args.dataset + '.yaml')

    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)
    with open(joint_train_config_path, 'r') as f:
        joint_train_config = yaml.load(f, Loader=loader)


    model_configs = []
    models = args.models.split(',')
    for model in models:
        model_config_path = os.path.join('configs/model_configs', model + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)
            model_configs += [model_config[args.dataset]]
    
    init_seed(joint_train_config['seed'], joint_train_config['reproducibility'])
    recsys = RecSys(model_configs, data_config, joint_train_config)
    recsys_evals = recsys.test_all_stages('test')

    for stage in recsys_evals:
        eval_output = set_color('eval result of stage: {}'.format(stage), 'blue') \
                        + ': \n' + dict2str(recsys_evals[stage])
        print(eval_output)
    
    # recsys.gen_stage_data('train')
