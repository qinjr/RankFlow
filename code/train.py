import argparse
import os
import yaml

from recbole.utils import init_seed
from utils.log import init_logger
from utils.yaml_loader import get_yaml_loader
from logging import getLogger
from joint_trainer import jointTrainer
from recsys import RecSys


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

    run_config = {'model': args.models,
                  'dataset': args.dataset,
                  'train_phase': 'joint'}

    init_seed(joint_train_config['seed'], joint_train_config['reproducibility'])
    # logger initialization
    init_logger(run_config)
    logger = getLogger()

    logger.info('training begin...')
    logger.info(run_config)
    logger.info(joint_train_config)

    model_configs = []
    models = args.models.split(',')
    for model in models:
        model_config_path = os.path.join('configs/model_configs', model + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)
            model_configs += [model_config[args.dataset]]


    recsys = RecSys(model_configs, data_config, joint_train_config, load_models = True)
    joint_trainer = jointTrainer(joint_train_config, recsys, run_config['dataset'])
    joint_trainer.joint_training()


    # '''for tuning'''
    # for lr in [1e-4, 1e-5]:
    #     recsys = RecSys(model_configs, data_config, joint_train_config)
    #     lr_dict = {}
    #     for i, stage in enumerate(joint_train_config['stage_names']):
    #         lr_dict[stage] = lr
    #     joint_train_config['learning_rates'] = lr_dict

    #     logger = getLogger()
    #     logger.info('\n\n\ntraining begin...with lr = {}'.format(lr))
    #     logger.info(joint_train_config)
        
    #     joint_trainer = jointTrainer(joint_train_config, recsys, run_config['dataset'])
    #     joint_trainer.joint_training()

