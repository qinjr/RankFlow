import argparse
import os
import yaml

# from torch.utils.data import DataLoader
from recbole.utils import init_seed
from dataloader import Dataloader
from dataset import IndDataset
from utils.log import init_logger
from utils.yaml_loader import get_yaml_loader
from logging import getLogger
from models import LR, WDL, DSSM, FM, DeepFM, YouTubeDNN, DIN, COLD, DIEN
from ind_trainer import indTrainer

def get_model(model_name, model_config, data_config):
    model_name = model_name.lower()
    if model_name == 'lr':
        return LR(model_config, data_config)
    elif model_name == 'dssm':
        return DSSM(model_config, data_config)
    elif model_name == 'wdl':
        return WDL(model_config, data_config)
    elif model_name == 'fm':
        return FM(model_config, data_config)
    elif model_name == 'deepfm':
        return DeepFM(model_config, data_config)
    elif model_name == 'youtubednn':
        return YouTubeDNN(model_config, data_config)
    elif model_name == 'din':
        return DIN(model_config, data_config)
    elif model_name == 'cold':
        return COLD(model_config, data_config)
    elif model_name == 'dien':
        return DIEN(model_config, data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='model name', default='lr')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ml-1m')
    args = parser.parse_args()

    if args.model in ['dssm'] and args.dataset == 'tiangong':
        random_neg_sample = True
    else:
        random_neg_sample = False

    # go to root path
    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    train_config_path = os.path.join('configs/ind_train_configs', args.dataset + '.yaml')
    model_config_path = os.path.join('configs/model_configs', args.model + '.yaml')

    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)
    with open(train_config_path, 'r') as f:
        train_config = yaml.load(f, Loader=loader)
    with open(model_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=loader)[args.dataset]

    run_config = {'model': args.model,
                  'dataset': args.dataset,
                  'train_phase': 'ind'}
    init_seed(train_config['seed'], train_config['reproducibility'])
    # logger initialization
    init_logger(run_config)
    logger = getLogger()

    # for lr in [1e-4]: #[1e-5, 1e-4, 1e-3]:
    #     for wd in [1e-4]:
    #         train_config['learning_rate'] = lr
    #         train_config['weight_decay'] = wd

    #         logger.info('\n\ntraining begin...with lr={}'.format(lr))
    #         logger.info(run_config)
    logger.info(train_config)

    # datasets: train and test
    train_dataset = IndDataset(data_config, 'train', train_config['train_mode'], random_neg_sample)
    if train_config['eval_mode'] == 'all':
        test_dataset = IndDataset(data_config, 'test', 'list', False)
    else:
        test_dataset = IndDataset(data_config, 'test', train_config['eval_mode'], False)

    # dataloaders: train and test
    train_tuple = train_dataset.get_data()
    
    if 'batch_random_neg' in model_config and model_config['batch_random_neg']:
        if train_config['train_mode'] == 'point':
            x_user, x_item, y = train_tuple
            train_tuple[0] = x_user[y == 1]
            train_tuple[1] = x_item[y == 1]
            train_tuple[2] = y[y == 1]

    train_dl = Dataloader(train_tuple, train_config['train_batch_size'], shuffle=True)
    test_dl = Dataloader(test_dataset.get_data(), train_config['eval_batch_size'], shuffle=False)

    # get model
    model = get_model(run_config['model'], model_config, data_config).to(train_config['device'])
    logger.info(model)
    # get trainer and fit
    trainer = indTrainer(train_config, model, args.dataset)
    best_eval_result = trainer.fit(train_dl, test_dl)


    # # load best model and test it
    # logger.info('Loading the best model and test...')
    # test_dl.refresh()
    # load_eval_result = trainer.evaluate(test_dl)
    # load_eval_output = set_color('loaded model\'s eval result', 'blue') + ': \n' + dict2str(load_eval_result)
    # logger.info(load_eval_output)
