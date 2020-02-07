import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import matplotlib as mpl
import time
import tensorflow as tf

# mpl.rcParams['font.sans-serif'] = ['SimHei']

data_dir = "D:\\qqfile\\data\\"
os.chdir(data_dir)
file_chdir = os.getcwd()


def my_date_range(begin, end, time_delta=1):
    dates = []
    dt = datetime.datetime.strptime(begin, "%Y-%m-%d")
    my_date = begin[:]
    while my_date <= end:
        dates.append(my_date)
        dt = dt + datetime.timedelta(time_delta)
        my_date = dt.strftime("%Y-%m-%d")
    return dates


def init_data_frame(data_frame, file_list):
    for f in file_list:
        data_frame = data_frame.append(pd.read_table(f, header=None, encoding='gb2312',
                                                     sep='\\]\\[|\\[|\\]', engine='python'))
    data_frame.drop(0, axis=1, inplace=True)
    data_frame.drop(45, axis=1, inplace=True)
    data_frame = data_frame.sort_values(1, 'index')
    return data_frame.reset_index(drop=True)


def add_data_to_result(temp, date_list, res, y_value):
    # row_of_data = temp.mean().round(2)  # avg
    row_of_data = temp.quantile().round(2)  # middle
    inner_s = pd.Series([date_list[-1], y_value], index=[1, 45])
    row_of_data = row_of_data.append(inner_s)
    row_of_data = row_of_data.sort_index(axis=0)
    return res.append(row_of_data, ignore_index=True)


def trans_data(data_frame, file_list, res_df, y_value):
    data_frame = init_data_frame(data_frame, file_list)
    print('data frame inited ' + str(y_value), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    date_list = []
    inner_temp = pd.DataFrame()

    for r_index, r in data_frame.iterrows():
        if inner_temp.empty:
            date_list.append(r[1][0:10])
            r.drop(labels=1, inplace=True)
            inner_temp = inner_temp.append(r, ignore_index=True)
        else:
            if r[1][0:10] != date_list[-1]:
                res_df = add_data_to_result(inner_temp, date_list, res_df, y_value)
                inner_temp = pd.DataFrame()
                date_list.append(r[1][0:10])
            else:
                r.drop(labels=1, inplace=True)
                inner_temp = inner_temp.append(r, ignore_index=True)
                if len(data_frame) - 1 == r_index:
                    res_df = add_data_to_result(inner_temp, date_list, res_df, y_value)
    return res_df


def get_result_df(cwd, date_range, tag):
    files0 = []
    files1 = []
    files2 = []
    for root, dirs, files in os.walk(cwd):
        for name in files:
            path = os.path.dirname(root)
            # if path == data_dir + "2014\\0":
            for d in date_range:
                if root == data_dir + d.split('-')[0] + "\\0\\" + d:
                    # if path == data_dir + "2012\\0":
                    files0.append(os.path.join(root, name))
                elif root == data_dir + d.split('-')[0] + "\\1\\" + d:
                    files1.append(os.path.join(root, name))
                # elif path == data_dir + "2012\\2":
                elif root == data_dir + d.split('-')[0] + "\\2\\" + d:
                    files2.append(os.path.join(root, name))
                else:
                    continue
    print(tag + ' files read finished time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    df = pd.DataFrame()
    result_df = pd.DataFrame()
    result_df = trans_data(df, files0, result_df, 0)
    result_df = trans_data(df, files1, result_df, 1)
    result_df = trans_data(df, files2, result_df, 2)
    print(len(df), tag)
    return result_df


# add the layer of neural network
def add_layer(inputs, in_size, out_size, activation_function=None):
    w = tf.Variable(tf.zeros([in_size, out_size]) + 0.01)  # 定义，in_size行,out_size列的矩阵,随机矩阵，全为0效果不佳
    b = tf.Variable(tf.zeros([1, out_size]) + 0.01)  # 不建议为0
    wx_plus_b = tf.matmul(inputs, w) + b  # wx + b
    if activation_function is None:
        output = wx_plus_b
    else:
        output = activation_function(wx_plus_b)
    return output


date_range_train = my_date_range('2012-01-01', '2014-12-31', 10)
train_df = get_result_df(file_chdir, date_range_train, "train")
date_range_test = my_date_range('2015-01-01', '2015-12-31', 10)
test_df = get_result_df(file_chdir, date_range_test, "test")

# X = tf.placeholder(tf.float32,[None,7])
# Y = tf.placeholder(tf.float32,[None,1])

x = tf.keras.Input(name="x", shape=(None, 44), dtype=tf.dtypes.float32)
y = tf.keras.Input(name="y", shape=(None, 1), dtype=tf.dtypes.float32)

output1 = add_layer(x, 44, 89, activation_function=tf.nn.sigmoid)
output2 = add_layer(output1, 89, 44, activation_function=tf.nn.sigmoid)
temp_y = add_layer(output2, 44, 1, activation_function=tf.nn.sigmoid)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - temp_y), reduction_indices=[1]))  # 先求平方，再求和，在求平均

train_step = tf.optimizers.Adam(0.004).minimize(loss)
