# RankFlow
This is the `PyTorch` implementation of our SIGIR 2022 paper [RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows](https://dl.acm.org/doi/abs/10.1145/3477495.3532050).


## Data Preparation & Preprocessing
The raw datasets are [ML-1M](https://grouplens.org/datasets/movielens/1m/), [TianGong-ST](http://www.thuir.cn/tiangong-st/), [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42).

The downloaded raw data should be placed into `[dataset_name]/raw_data` folder. And all the preprocessed data will be placed at `[dataset_name]/feateng_data` folder. To execute the data processing procedure, just use
```
python process_[dataset_name].py
```
in the corresponding directory.


## Warmup Training (Independent Training Baseline)
To execute the independent training on the impression (displayed) data, use
```
python warmup.py -d [dataset_name] -m [model_name]
```
or use the shell scripts (to train multiple models) as
```
bash ind_train.sh -d [dataset_name] -m [list of model_names]
```

## RankFlow Training
To execute the RankFlow joint training, use
```
python train.py -d [dataset_name] -ms [list of model_names in the cascade]
```

## Citation
```
@inproceedings{qin2022rankflow,
  title={RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows},
  author={Qin, Jiarui and Zhu, Jiachen and Chen, Bo and Liu, Zhirong and Liu, Weiwen and Tang, Ruiming and Zhang, Rui and Yu, Yong and Zhang, Weinan},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={814--824},
  year={2022}
}
```