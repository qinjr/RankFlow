import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

THRESHOLD = 4

def join_files(rating_file, movie_file, user_file, joined_file):
    print('loading datasets...')
    rating_df = pd.read_csv(rating_file, sep='::', names=['uid', 'iid', 'rating', 'time'], encoding='unicode_escape')[['uid','iid','rating']]
    movie_df = pd.read_csv(movie_file, sep='::', names=['iid', 'title', 'cid'], encoding='unicode_escape')[['iid', 'cid']]
    movie_df['cid'] = movie_df['cid'].map(lambda x:x.split('|')[0])
    user_df = pd.read_csv(user_file, sep='::', names=['uid', 'gid', 'aid', 'oid', 'zipcode'], encoding='unicode_escape')
    print('datasets loaded')

    print(rating_df)
    print(movie_df)
    print(user_df)

    # join
    join_df = pd.merge(pd.merge(rating_df, movie_df, on='iid'), user_df, on='uid')
    join_df['rating'] = join_df['rating'].map(lambda x: 1 if x > THRESHOLD else 0)
    join_df = join_df[['uid', 'gid', 'aid', 'oid', 'zipcode', 'iid', 'cid', 'rating']]
    print(join_df)
    print('joined completed')

    # write to csv
    join_df.to_csv(joined_file, header=True, index=False)


def feature_remap(joined_file, remapped_file):
    join_df = pd.read_csv(joined_file)
    print(join_df)

    lbe = LabelEncoder()
    col_names = ['uid', 'gid', 'aid', 'oid', 'zipcode', 'iid', 'cid']
    for col in col_names:
        join_df[col] = lbe.fit_transform(join_df[col])
    print(join_df)

    feat_size = 0
    for col in col_names:
        join_df[col] = join_df[col].map(lambda x:x+feat_size)
        feat_size += len(join_df[col].unique())
    print(join_df)
    print(feat_size)
    
    join_df.to_csv(remapped_file, header=True, index=False)


def split(remapped_file, train_log_file, unpv_log_file, test_log_file):
    data_df = pd.read_csv(remapped_file)
    train_log_df, test_log_df = train_test_split(data_df, test_size=0.2)
    # 1:1 split pv and unpv data
    train_log_df, unpv_log_df = train_test_split(train_log_df, test_size=0.5)

    train_log_df.to_csv(train_log_file, header=True, index=False)
    unpv_log_df.to_csv(unpv_log_file, header=True, index=False)
    test_log_df.to_csv(test_log_file, header=True, index=False)
    print(train_log_df)
    print(unpv_log_df)
    print(test_log_df)

    print('dataset splitting completed')

def gen_feat_dfs(remapped_file, user_df_file, item_df_file):
    table_df = pd.read_csv(remapped_file)

    table_df_user = table_df.drop_duplicates('uid')[['uid', 'gid', 'aid', 'oid', 'zipcode']]
    table_df_item = table_df.drop_duplicates('iid')[['iid', 'cid']]
    
    # user_feat_dict = table_df_user.set_index('uid').T.to_dict('list')
    # item_feat_dict = table_df_item.set_index('iid').T.to_dict('list')
    

    table_df_user.to_csv(user_df_file, header=True, index=False)
    table_df_item.to_csv(item_df_file, header=True, index=False)
    
    print('feature files for users and items are generated')


if __name__ == '__main__':
    join_files('raw_data/ratings.dat', 'raw_data/movies.dat', 'raw_data/users.dat', 'feateng_data/joined_raw.csv')
    feature_remap('feateng_data/joined_raw.csv', 'feateng_data/all.inter')
    split('feateng_data/all.inter', 'feateng_data/train.inter', 'feateng_data/unpv.inter', 'feateng_data/test.inter')
    gen_feat_dfs('feateng_data/all.inter', 'feateng_data/feat.user', 'feateng_data/feat.item')
