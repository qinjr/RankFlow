import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from multiprocessing import Pool
from functools import partial
import time

THRESHOLD = 4
with open("feateng_data/cat_item_dict.pkl", 'rb') as f:
    cat_item_dict = pkl.load(f)
with open("feateng_data/items.pkl", 'rb') as f:
    items = pkl.load(f)
with open("feateng_data/item_dict.pkl", 'rb') as f:
    item_dict = pkl.load(f)


def join_files(info_file, log_file, joined_file):
    print('loading datasets...')
    info_df = pd.read_csv(info_file, header = 0 ,names =['uid','aid','gid'], keep_default_na=False)
    log_df = pd.read_csv(log_file, header = 0, names = ['uid','iid','cid','sid','bid','time','action'], keep_default_na=False)

    print("dataset loaded")

    log_df = log_df.replace("",-1)
    info_df = info_df.replace("",-1)

    log_df = log_df[log_df["action"] == 0]
    log_df = log_df.loc[:, "uid":"time"]

    print(log_df)
    print(info_df)

    # join
    join_df = pd.merge(log_df, info_df, on='uid')

    print(join_df)
    print('joined completed')

    # write to csv
    join_df.to_csv(joined_file, header=True, index=False)


def feature_remap(joined_file, remapped_file):
    join_df = pd.read_csv(joined_file)
    print(join_df)

    lbe = LabelEncoder()
    col_names = ['uid', 'aid', 'gid', 'iid', 'cid', 'sid','bid']
    for col in col_names:
        join_df[col] = lbe.fit_transform(join_df[col])
    print(join_df)

    feat_size = 1 #0 is used to pad.
    for col in col_names:
        join_df[col] = join_df[col].map(lambda x: x + feat_size)
        feat_size += len(join_df[col].unique())
    print(join_df)
    print(feat_size-1)

    join_df.to_csv(remapped_file, header=True, index=False)


def split(remapped_file, train_log_file, unpv_log_file, test_log_file, feat_user, feat_item):
    data_df = pd.read_csv(remapped_file)
    user_df = pd.read_csv(feat_user)
    item_df = pd.read_csv(feat_item)

    data_df = data_df.sort_values(by =['time'])

    length = len(data_df)

    history_df = data_df[0:int(0.4*length)]
    train_log_df = data_df[int(0.4*length):int(0.9*length)]
    unpv_log_df = data_df[int(0.9*length):int(0.95*length)]
    test_log_df = data_df[int(0.95*length):length]

    print(train_log_df)

    #history:
    def process_hist():
        hist_dict = {}
        def dict_init(x):
            hist_dict[x['uid']] = []
        user_df.apply(dict_init,axis=1)
        print(hist_dict)

        def func0(x):
            hist_dict[x['uid']].append(x['iid'])
        history_df.apply(func0,axis=1)
        print(hist_dict)

        padded_length = 20
        for key, item in hist_dict.items():
            if len(item)>padded_length:
                del item[0:len(item)-padded_length]
            else:
                for i in range(padded_length - len(item)):
                    item.append(0)

        print("history is generated")
        with open("feateng_data/hist_dict.pkl", "wb") as fw:
            pkl.dump(hist_dict, fw)
    # process_hist()

    #############################################
    #cat item dict
    items = np.array(item_df['iid'])
    with open("feateng_data/items.pkl",'wb') as f:
        pkl.dump(items,f)
    cat_item_dict = {}
    item_dict={}
    def dict_init():
        def cat_dict_init(x):
            cat_item_dict[x['cid']] = []
        item_df.apply(cat_dict_init, axis=1)

        def func1(x):
            cat_item_dict[x['cid']].append(x['iid'])
        item_df.apply(func1, axis=1)

        with open("feateng_data/cat_item_dict.pkl",'wb') as f:
            pkl.dump(cat_item_dict,f)

        def item_dict_init(x):
            item_dict[x['iid']] = [x['cid'], x['sid'],x['bid']]
        item_df.apply(item_dict_init, axis=1)
        with open("feateng_data/item_dict.pkl",'wb') as f:
            pkl.dump(item_dict,f)
    #dict_init()
    #train
    def train_unpv_test_generate(log_df,file_name):
        def parallelize_on_rows(data, func1,func2, num_of_processes=5):
            return parallelize(data, partial(run_on_subset, func1),partial(run_on_subset, func2), num_of_processes)

        #生成train_file并储存

        log_df = log_df[['uid', 'aid','gid','iid', 'cid', 'sid','bid']]
        log_df = parallelize_on_rows(log_df, negative1, negative2, 8)
        log_df.replace(np.nan,1)
        log_df.to_csv(file_name, index=False)
    #
    train_unpv_test_generate(train_log_df,train_log_file)
    train_unpv_test_generate(unpv_log_df,unpv_log_file)
    train_unpv_test_generate(test_log_df,test_log_file)

    print('dataset splitting completed')


def run_on_subset(func, data_subset):
    global cat_item_dict
    return data_subset.apply(func, axis=1)

def parallelize(data, func1, func2, num_of_processes=5):
    global cat_item_dict
    data_split = np.array_split(data, num_of_processes)
    data['ranking'] = 1
    pool = Pool(num_of_processes)
    data1 = pd.concat(pool.map(func1, data_split), axis=0)
    data2 = pd.concat(pool.map(func2, data_split), axis=0)
    data = pd.concat([data, data1, data2], axis=0)
    pool.close()
    pool.join()
    return data

def negative1(row):
    # 添加计算操作
    global cat_item_dict
    global items
    global item_dict
    this_item = row['iid']
    items_in_this_cat = cat_item_dict[item_dict[this_item][0]].copy()
    items_in_this_cat.remove(this_item)
    if len(items_in_this_cat) == 0:
        item_negative1 = random.choice(items)
        while item_negative1 == this_item:
            item_negative1 = random.choice(items)
    else:
        item_negative1 = random.choice(items_in_this_cat)

    return pd.Series([row['uid'], row['aid'],row['gid'],item_negative1, item_dict[item_negative1][0], item_dict[item_negative1][1], item_dict[item_negative1][2], int(0)],
                     index=['uid','aid','gid', 'iid', 'cid', 'sid','bid' ,'ranking'])


def negative2(row):
    # 添加计算操作
    global cat_item_dict
    global items
    global item_dict
    this_item = row['iid']
    item_negative2 = random.choice(items)
    while item_negative2 == this_item:
        item_negative2 = random.choice(items)
    return pd.Series([row['uid'], row['aid'],row['gid'],item_negative2, item_dict[item_negative2][0], item_dict[item_negative2][1],
                      item_dict[item_negative2][2], int(0)],
                     index=['uid', 'aid','gid','iid', 'cid', 'sid', 'bid', 'ranking'])


def gen_feat_dfs(remapped_file, user_df_file, item_df_file):
    table_df = pd.read_csv(remapped_file)

    user = table_df['uid'].unique()


    table_df_user = table_df.drop_duplicates('uid')[['uid','aid','gid']]
    table_df_item = table_df.drop_duplicates('iid')[['iid','cid', 'sid','bid']]

    #user_feat_dict = table_df_user.set_index('uid').T.to_dict('list')
    # item_feat_dict = table_df_item.set_index('iid').T.to_dict('list')


    table_df_user.to_csv(user_df_file, header=True, index=False)
    table_df_item.to_csv(item_df_file, header=True, index=False)

    print('feature files for users and items are generated')


if __name__ == '__main__':
    join_files('raw_data/user_info_format1.csv','raw_data/user_log_format1.csv', 'feateng_data/all.inter')
    feature_remap('feateng_data/all.inter', 'feateng_data/remapped.inter')
    gen_feat_dfs('feateng_data/remapped.inter', 'feateng_data/feat.user', 'feateng_data/feat.item')
    start = time.time()
    split('feateng_data/remapped.inter', 'feateng_data/train.inter', 'feateng_data/unpv.inter', 'feateng_data/test.inter','feateng_data/feat.user','feateng_data/feat.item')
    end = time.time()
    print(end-start)