import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import time

def generate_data(input_file,output_file):
    log_df = pd.read_csv(input_file, sep='\t', names=['a', 'qid', 'b', 'c', 'did', 'vid', 'click'], encoding='unicode_escape')[
        ['qid','did', 'vid', 'click']]

    print(type(log_df['did'][0]))
    def parallelize_on_rows(data, func1, i, num_of_processes=5):

        return parallelize(data, partial(run_on_subset, func1, i), num_of_processes)

    # 生成train_file并储存

    log_df = parallelize_on_rows(log_df, str_to_list,0, 16)
    df = []
    for i in range(10):
        df.append(parallelize_on_rows(log_df, open_list,i, 16))
    log_df = pd.concat(df)
    log_df.replace(np.nan, 1)
    log_df.to_csv(output_file, index=False)

def run_on_subset(func, i,data_subset):
    return data_subset.apply(func, axis=1,args=(i,))

def parallelize(data, func1, num_of_processes=5):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func1, data_split), axis=0)
    pool.close()
    pool.join()
    return data

def str_to_list(row,i):
    return pd.Series([row['qid'], row['did'][1:-1].split(', '), row['vid'][1:-1].split(', '), row['click'][1:-1].split(', ')],
                     index =['qid', 'did', 'vid', 'click'])

def open_list(row,i):
    # 添加计算操作

    return pd.Series([row['qid'], row['did'][i], row['vid'][i], row['click'][i]],
                     index=['qid', 'did', 'vid', 'click'])


def join_files(file1,file2,file3,query_df_file,doc_df_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    table_df = pd.concat([df1,df2,df3])

    feat_size = 1
    for col in ['qid','did','vid']:
        df1[col] = df1[col].map(lambda x: x + feat_size)
        df2[col] = df2[col].map(lambda x: x + feat_size)
        df3[col] = df3[col].map(lambda x: x + feat_size)
        table_df[col] = table_df[col].map(lambda x: x + feat_size)
        feat_size += table_df[col].unique().size
    print(feat_size)
    table_df_query = table_df.drop_duplicates('qid')['qid']
    table_df_doc = table_df.drop_duplicates('did')[['did', 'vid']]

    df1.to_csv(file1, header=True, index=False)
    df2.to_csv(file2, header=True, index=False)
    df3.to_csv(file3, header=True, index=False)
    

    table_df_query.to_csv(query_df_file, header=True, index=False)
    table_df_doc.to_csv(doc_df_file, header=True, index=False)


if __name__ == '__main__':
    start = time.time()
    #generate_data('feateng_data/train_per_query.txt', 'feateng_data/train.inter')
    #generate_data('feateng_data/dev_per_query.txt', 'feateng_data/dev.inter')
    #generate_data('feateng_data/test_per_query.txt', 'feateng_data/test.inter')
    join_files('feateng_data/train.inter','feateng_data/dev.inter','feateng_data/test.inter','feateng_data/feat.query','feateng_data/feat.doc')
    
    end = time.time()
    print(end-start)
