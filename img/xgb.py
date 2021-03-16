# -*- coding: utf-8 -*-
"""
@Path: 
@Author: zhongzhilai
@Email: zhongzhilai001@ke.com
@Date: 2021/2/2 18:17
@Description: 
"""
import math
import re
from math import radians, sin, cos, atan2, sqrt

import random
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer, precision_score, accuracy_score, recall_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, ShuffleSplit
import pickle
from sklearn.preprocessing import KBinsDiscretizer

div_num = 15


def load_data():
    data_train_0 = pd.read_csv("D:/zhongzhilai001/Desktop/typing_length/shanghai_20200901-0930.csv", header=0,
                               index_col=False)
    data_train_1 = pd.read_csv("D:/zhongzhilai001/Desktop/typing_length/shanghai_20201001-1030.csv", header=0,
                               index_col=False)
    data_train_2 = pd.read_csv("D:/zhongzhilai001/Desktop/typing_length/shanghai_20201101-1130.csv", header=0,
                               index_col=False)
    data_train = pd.concat([data_train_0, data_train_1, data_train_2], axis=0, join='outer', ignore_index=True)
    data_test = pd.read_csv("D:/zhongzhilai001/Desktop/typing_length/shanghai_20201201-1230.csv", header=0,
                            index_col=False)
    return data_train, data_test


def clean_data(data_train, data_test):
    data_train.drop(['house_id', 'housedel_id', 'unit_id', 'resblock_id', 'frame_id', 'agreement_amt'], axis=1,
                    inplace=True)

    # data_train.pop('reference_amt')
    # data_train.pop('resblock_reference_amt')
    # data_train.pop('typing_time')

    data_test.drop(['house_id', 'housedel_id', 'unit_id', 'resblock_id', 'frame_id', 'agreement_amt'], axis=1,
                   inplace=True)
    # data_test.pop('reference_amt')
    # data_test.pop('resblock_reference_amt')
    # data_test.pop('typing_time')

    # 空字段填充为nan
    data_train['flying_dist'] = data_train['flying_dist'].apply(lambda x: np.nan if x == 0 else x)
    data_train['building_type_code'] = data_train['building_type_code'].apply(lambda x: np.nan if x == -911 else x)
    data_train['frame_structure_code'] = data_train['frame_structure_code'].apply(lambda x: np.nan if x == -911 else x)
    data_train['parking_type_code'] = data_train['parking_type_code'].apply(lambda x: np.nan if x == -911 else x)
    data_train['parking_ratio'] = data_train['parking_ratio'].apply(lambda x: np.nan if x == -1 else x)

    data_test['flying_dist'] = data_test['flying_dist'].apply(lambda x: np.nan if x == 0 else x)
    data_test['building_type_code'] = data_test['building_type_code'].apply(lambda x: np.nan if x == -911 else x)
    data_test['frame_structure_code'] = data_test['frame_structure_code'].apply(lambda x: np.nan if x == -911 else x)
    data_test['parking_type_code'] = data_test['parking_type_code'].apply(lambda x: np.nan if x == -911 else x)
    data_test['parking_ratio'] = data_test['parking_ratio'].apply(lambda x: np.nan if x == -1 else x)

    # 去除价格变化数据
    data_train = data_train[(~data_train['typing_length'].str.contains(','))]
    data_test = data_test[(~data_test['typing_length'].str.contains(','))]

    data_train.pop('agreement_amt')
    data_test.pop('agreement_amt')

    # data_test = data_test[(data_test['typing_time'].isin([11, 12]))]
    # data_train.pop('typing_time')
    # data_test.pop('typing_time')

    # 去除重复行
    data_train.drop_duplicates(subset=None, keep='first', inplace=True)
    data_test.drop_duplicates(subset=None, keep='first', inplace=True)

    # 去除label为空数据
    data_train.dropna(subset=['typing_length'], inplace=True)
    data_test.dropna(subset=['typing_length'], inplace=True)

    # object类型转int,float
    data_train['typing_length'] = data_train['typing_length'].astype(int)
    data_train['monthly_rent_amt'] = data_train['monthly_rent_amt'].astype(float)

    data_test['typing_length'] = data_test['typing_length'].astype(int)
    data_test['monthly_rent_amt'] = data_test['monthly_rent_amt'].astype(float)

    # 去除离群数据
    data_train = data_train[(data_train['typing_length'] >= 1) & (data_train['typing_length'] <= 90)]
    data_test = data_test[(data_test['typing_length'] >= 1) & (data_test['typing_length'] <= 90)]

    # label分类
    # data_train['typing_length'] = data_train['typing_length'].apply(lambda x: 0 if x < 7 else (1 if x < 15 else (2 if x < 30 else 3)))
    # data_test['typing_length'] = data_test['typing_length'].apply(lambda x: 0 if x < 7 else (1 if x < 15 else (2 if x < 30 else 3)))

    # data_train['typing_length'] = data_train['typing_length'].apply(lambda x: 0 if x < 10 else (1 if x < 30 else 2))
    # data_test['typing_length'] = data_test['typing_length'].apply(lambda x: 0 if x < 10 else (1 if x < 30 else 2))

    # data_train['typing_length'] = data_train['typing_length'].apply(lambda x: 0 if x < div_num else 1)
    # data_test['typing_length'] = data_test['typing_length'].apply(lambda x: 0 if x < div_num else 1)

    # print '0:', data_train['typing_length'].apply(lambda x: 1 if x <= 0 else 0).sum()
    # print '1:', data_train['typing_length'].apply(lambda x: 1 if 0 < x <= 1 else 0).sum()
    # print '2:', data_train['typing_length'].apply(lambda x: 1 if 1 < x <= 2 else 0).sum()
    # print '3:', data_train['typing_length'].apply(lambda x: 1 if 2 < x <= 3 else 0).sum()

    # 新增字段
    # data_train['amt_diff'] = data_train['monthly_rent_amt'] - data_train['agreement_amt']
    # data_train['amt_diff_ratio'] = (data_train['monthly_rent_amt'] - data_train['agreement_amt']) / data_train['agreement_amt']
    # data_test['amt_diff'] = data_test['monthly_rent_amt'] - data_test['agreement_amt']
    # data_test['amt_diff_ratio'] = (data_test['monthly_rent_amt'] - data_test['agreement_amt']) / data_test['agreement_amt']
    data_train['floor_ratio'] = data_train['physical_floor'] / data_train['overground_floor_cnt']
    data_test['floor_ratio'] = data_test['physical_floor'] / data_test['overground_floor_cnt']

    # one-hot处理离散数据
    data_train = pd.get_dummies(data_train,
                                columns=['grade', 'face', 'fitment', 'building_type_code', 'parking_type_code',
                                         'frame_structure_code'])  # grad:房源等级编码 face:朝向, fitment:装修, parking_type_code:车位类型, frame_structure_code:户型结构编码 , building_type_code:建筑类型
    data_test = pd.get_dummies(data_test,
                               columns=['grade', 'face', 'fitment', 'building_type_code', 'parking_type_code',
                                        'frame_structure_code'])  # grad:房源等级编码 face:朝向, fitment:装修, parking_type_code:车位类型, frame_structure_code:户型结构编码 , building_type_code:建筑类型

    # 对齐训练测试集one-hot后的枚举值
    data_train = data_train.reindex(columns=list(set(data_train.columns).union(set(data_test.columns))), fill_value=0)
    data_test = data_test.reindex(columns=list(data_train.columns), fill_value=0)

    # 打乱数据行
    data_train = data_train.reindex(np.random.permutation(data_train.index))
    return data_train, data_test


def clean_data_v2(data_train, data_test):
    data_train.drop(
        ['house_id', 'housedel_id', 'unit_id', 'resblock_id', 'frame_id', 'agreement_amt', 'sign_time', 'typing_time'],
        axis=1, inplace=True)
    data_test.drop(
        ['house_id', 'housedel_id', 'unit_id', 'resblock_id', 'frame_id', 'agreement_amt', 'sign_time', 'typing_time'],
        axis=1, inplace=True)
    # -911替换为null
    for columns in ['fitment_status_code', 'grade_code', 'flying_dist', 'frame_face_code', 'build_year']:
        data_train.loc[data_train[columns] == '-911'] = np.nan
        data_test.loc[data_test[columns] == '-911'] = np.nan
    # 特殊字段替换为null
    data_train.loc[data_train['parking_cnt'] == '-2'] = np.nan
    data_train.loc[data_train['parking_ratio'] == '-1'] = np.nan
    data_test.loc[data_test['parking_cnt'] == '-2'] = np.nan
    data_test.loc[data_test['parking_ratio'] == '-1'] = np.nan
    # label等频分箱
    # data_train, data_test = data_qcut(data_train, data_test)
    # 去除多楼层
    data_train['floor_height_formatted'] = data_train['floor_height_formatted'].apply(
        lambda x: float(re.match('\d:(\d+)', str(x))[1]) / 1000.0 if re.match('\d:(\d+)', str(x)) else np.nan)
    data_test['floor_height_formatted'] = data_test['floor_height_formatted'].apply(
        lambda x: float(re.match('\d:(\d+)', str(x))[1]) / 1000.0 if re.match('\d:(\d+)', str(x)) else np.nan)
    # 去除某些字段为空的行
    # data_train.dropna(subset=['resblock_reference_amt'], inplace=True)
    # data_test.dropna(subset=['resblock_reference_amt'], inplace=True)
    # 去除离群数据
    data_train = data_train[(data_train['typing_length'] >= 0) & (data_train['typing_length'] <= 45)]
    data_train = data_train[(data_train['price'] <= 20000)]
    data_test = data_test[(data_test['typing_length'] >= 0) & (data_test['typing_length'] <= 45)]
    data_test = data_test[(data_test['price'] <= 20000)]
    # 新增字段
    data_train['floor_ratio'] = data_train['physical_floor'] / data_train['overground_floor_cnt']
    data_test['floor_ratio'] = data_test['physical_floor'] / data_test['overground_floor_cnt']
    data_train = deal_cell_area(data_train)
    data_test = deal_cell_area(data_test)
    # one-hot处理离散数据
    data_train = pd.get_dummies(data_train, columns=['fitment_status_code', 'grade_code', 'frame_face_code', ])
    data_test = pd.get_dummies(data_test, columns=['fitment_status_code', 'grade_code', 'frame_face_code', ])
    # 对齐训练测试集one-hot后的枚举值
    data_train = data_train.reindex(columns=list(set(data_train.columns).union(set(data_test.columns))), fill_value=0)
    data_test = data_test.reindex(columns=list(data_train.columns), fill_value=0)
    # 打乱数据行
    data_train = data_train.reindex(np.random.permutation(data_train.index))
    return data_train, data_test


def deal_cell_area(data):
    for i, v in data['cell_area'].iteritems():
        code_type = {'110008010': 0, '-911': 0, '110008008': 0, '110008004': 0, '110008013': 0,
                     '110008011': 0, '110008009': 0, '110008005': 0, '110008003': 0, '110008002': 0,
                     '110008006': 0, '110008012': 0, '110008001': 0, '110008007': 0}
        if not pd.isna(v):
            codes = str.split(v, ':')
            for _ in range(int(len(codes) / 2)):
                code_type[codes[2 * _]] += float(codes[2 * _ + 1])
            for k, value in code_type.items():
                data.loc[i, 'cell_area_' + k] = value
    data.pop('cell_area')
    return data


def data_qcut(data_train, data_test):
    k = 10  # 设置分箱数
    est = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='quantile')
    est.fit(data_train[['typing_length']])
    Xt = est.transform(data_train[['typing_length']])
    # 将离散化后的数据与之前的离散数据拼起来
    data_train['typing_length'] = pd.DataFrame(Xt, columns=[['typing_length']])
    cc = est.bin_edges_[0]
    print('划分区间:', cc)

    def xxx(num):
        for i in range(len(cc) - 1):
            if cc[i] <= num < cc[i + 1]:
                return i
        return len(cc) - 2

    data_test['typing_length'] = data_test['typing_length'].apply(lambda x: xxx(x))
    return data_train, data_test


def cut_data(data_train, data_test):
    # columns = ['building_latitude', 'building_longitude', 'room_cnt', 'fitment_1', 'fitment_2', 'fitment_3',
    #            'monthly_rent_amt', 'resblock_reference_amt', 'resblock_cubage_ratio', 'reference_amt', 'reference_area',
    #            'physical_floor', 'floor_area', 'image_cnt', 'floor_ratio', 'typing_time', 'build_year', 'typing_length']
    columns = ['building_latitude', 'building_longitude', 'flying_dist', 'parking_cnt', 'floor_area', 'floor_ratio',
               'typing_length']
    return data_train[columns], data_test[columns]


def filter_near_data(data_train, data_test):
    data_train_near = pd.DataFrame(columns=data_train.columns)
    for index, data in data_train.iterrows():
        if haversine(data['building_longitude'], data['building_latitude'], data_test['building_longitude'],
                     data_test['building_latitude']) <= 1500 and abs(
            data['monthly_rent_amt'] - data_test['monthly_rent_amt']) <= 150 and data['room_cnt'] == data_test[
            'room_cnt']:
            data_train_near = data_train_near.append(data, ignore_index=True)

    data_test_select = pd.DataFrame(columns=data_train.columns).append(data_test, ignore_index=True)
    return data_train_near, data_test_select


def haversine(lng1, lat1, lng2, lat2):
    """
    根据经纬度计算距离，单位 m
    :param lng1: 经度 1
    :param lat1: 纬度 1
    :param lng2: 经度 2
    :param lat2: 纬度 2
    :return:
    """
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    a = sin((lng1 - lng2) / 2) ** 2 + cos(lng1) * cos(lng2) * (sin((lat1 - lat2) / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c * 1000


def show_model(model, dtrain, y_test):
    # 模型可视化
    y_pred = pd.Series(model.predict(dtrain))
    plt.figure(figsize=(20, 10), facecolor='w')
    ln_x_test = range(len(y_pred))

    plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'realDate')
    plt.plot(ln_x_test, y_pred, 'g-', lw=4, label=u'XGBoostDate')
    plt.xlabel(u'room_id')
    plt.ylabel(u'agreement_amt')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    xgb.plot_importance(model)
    plt.show()


def predict(data_train, data_test):
    # data,lable 分开
    label_train = data_train.pop('typing_length')
    label_test = data_test.pop('typing_length')
    data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_train, test_size=0.2,
                                                                        random_state=41)
    data_train.reset_index(inplace=True, drop=True)
    label_train.reset_index(inplace=True, drop=True)
    data_valid.reset_index(inplace=True, drop=True)
    label_valid.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)
    label_test.reset_index(inplace=True, drop=True)
    dtrain = xgb.DMatrix(data_train, label=label_train)
    dvalid = xgb.DMatrix(data_valid, label=label_valid)
    dtest = xgb.DMatrix(data_test, label=label_test)

    param = {
        # 'learning_rate': 0.1,
        'eta': 0.1,
        # 'n_estimators': 5,
        'max_depth': 5,
        'min_child_weight': 20,
        'gamma': 0.1,
        'reg_lambda': 1,
        'reg_alpha': 0,
        # 'subsample': 0.9,
        # 'colsample_bytree': 0.9,
        # 'objective': 'binary:logistic',
        # 'objective': 'multi:softmax',
        'num_class': 10,
        'objective': 'reg:squarederror',
        # 'objective': 'rank:pairwise',
        'scale_pos_weight': 1, }
    model = xgb.train(param, dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], num_boost_round=400,
                      early_stopping_rounds=10)

    # save
    pickle.dump(model, open("xgbRegression.pkl", "wb"))
    # load
    model = pickle.load(open("xgbRegression.pkl", "rb"))

    print('模型参数：', param)
    # 模型结果
    print('------------训练集---------------')
    y_pred = pd.Series(model.predict(dtrain))
    # model_valid1(y_pred, label_train)
    # model_valid2(y_pred, label_train)
    model_valid3_2(y_pred, label_train)
    # model_valid3_1(y_pred, label_train)
    # model_valid4(model, data_train, label_train)
    # model_valid5(model, data_train, label_train)

    print('------------验证集---------------')
    y_pred = pd.Series(model.predict(dvalid))
    # model_valid1(y_pred, label_valid)
    # model_valid2(y_pred, label_valid)
    model_valid3_2(y_pred, label_valid)
    # model_valid3_1(y_pred, label_valid)
    # model_valid4(model, data_valid, label_valid)
    # model_valid5(model, data_valid, label_valid)

    print('------------测试集---------------')
    y_pred = pd.Series(model.predict(dtest))
    # model_valid1(y_pred, label_test)
    # model_valid2(y_pred, label_test)
    model_valid3_2(y_pred, label_test)
    # model_valid3_1(y_pred, label_test)
    # model_valid4(model, data_test, label_test)
    # model_valid5(model, data_test, label_test)
    # 结果保存
    # data_test = pd.read_excel("D:/zhongzhilai001/Desktop/shanghai_210101-210310.xlsx", header=0, index_col=False)
    # # 去除多楼层
    # data_test['floor_height_formatted'] = data_test['floor_height_formatted'].apply(
    #     lambda x: float(re.match('\d:(\d+)', str(x))[1]) / 1000.0 if re.match('\d:(\d+)', str(x)) else np.nan)
    # # 去除某些字段为空的行
    # # data_test.dropna(subset=['resblock_reference_amt'], inplace=True)
    # # 去除离群数据
    # data_test = data_test[(data_test['typing_length'] >= 1) & (data_test['typing_length'] <= 45)]
    # data_test = data_test[(data_test['price'] <= 20000)]
    # # 新增字段
    # data_test['floor_ratio'] = data_test['physical_floor'] / data_test['overground_floor_cnt']
    # save_csv(data_test, y_pred, label_test)

    show_model(model, dtest, label_test)


def near_predict(data_train, data_test):
    data_train.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)

    param = {'learning_rate': 0.1,
             'n_estimators': 50,
             'max_depth': 5,
             'min_child_weight': 1,
             'gamma': 0.1,
             'reg_lambda': 2,
             'reg_alpha': 0,
             # 'subsample': 0.9,
             # 'colsample_bytree': 0.9,
             # 'objective': 'binary:logitraw',
             # 'objective': 'multi:softmax  num_class=3',
             'objective': 'reg:logistic',
             'scale_pos_weight': 1, }
    model = XGBClassifier(**param)
    total = len(data_test)
    rights = [0.0, 0.0]
    totals = [0.0, 0.0]
    recall_totals = [0.0, 0.0]
    jindu = 0.0
    MSE = 0.0
    for index, x_test_row in data_test.iterrows():
        x_train_near, x_test_select = filter_near_data(data_train, x_test_row)
        y_train_near = x_train_near.pop('typing_length')
        y_test_row = x_test_select.pop('typing_length')
        if len(x_train_near) >= 20:
            model.fit(x_train_near, y_train_near)
            y_pred = pd.Series(model.predict(x_test_select))
            # 分类
            # rights[0] += 1 if (y_pred[0] == y_test_row[0]) else 0
            # 回归
            # rights[0] += 1 if abs(y_pred - y_test_row)[0] <= 3 else 0
            # rights[1] += 1 if abs(y_pred - y_test_row)[0] <= 5 else 0
            # MSE += abs(y_pred - y_test_row)[0]
            # 回归 手动分类
            rights[0] += 1 if y_pred[0] <= 15 and y_test_row[0] <= 15 else 0
            rights[1] += 1 if y_pred[0] > 15 and y_test_row[0] > 15 else 0
            totals[0] += 1 if y_pred[0] <= 15 else 0
            totals[1] += 1 if y_pred[0] > 15 else 0
            recall_totals[0] += 1 if y_test_row[0] <= 15 else 0
            recall_totals[1] += 1 if y_test_row[0] > 15 else 0
            print('pred：{}, label：{}'.format(y_pred[0], y_test_row[0]))

            print('[0,15)准确率：{} ({} / {})'.format(rights[0] / totals[0] if totals[0] != 0 else 0, rights[0], totals[0]))
            print(
                '[15,60]准确率：{} ({} / {})'.format(rights[1] / totals[1] if totals[1] != 0 else 0, rights[1], totals[1]))
            print(
                '[0,15)召回率：{} ({} / {})'.format(rights[0] / recall_totals[0] if recall_totals[0] != 0 else 0, rights[0],
                                                recall_totals[0]))
            print('[15,60]召回率：{} ({} / {})'.format(rights[1] / recall_totals[1] if recall_totals[1] != 0 else 0,
                                                   rights[1], recall_totals[1]))
            # print ('平均误差：{}'.format(MSE / totals))

        if jindu % 100 == 0:
            print('进度：{}, {}'.format(jindu / 100, jindu / total))
        jindu += 1


def select_feature_model(data_test, data_train, label_test, label_train, model, param):
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(data_train)
        # train model
        selection_model = XGBClassifier(**param)
        selection_model.fit(select_X_train, label_train)
        # eval model
        select_X_test = selection.transform(data_test)
        y_pred = selection_model.predict(select_X_test)
        accuracy = accuracy_score(label_test, y_pred)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
                                                       accuracy * 100.0))


def aft_predict(df_train, df_test):
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    y_lower_bound = df_train['typing_length']
    y_upper_bound = df_train['typing_length']
    X = df_train.drop(['typing_length'], axis=1)
    test_y_lower_bound = df_test['typing_length']
    test_y_upper_bound = df_test['typing_length']
    X_test = df_test.drop(['typing_length'], axis=1)

    rs = ShuffleSplit(n_splits=2, test_size=.7, random_state=0)
    train_index, valid_index = next(rs.split(X))
    dtrain = xgb.DMatrix(X.values[train_index, :])
    dtrain.set_float_info('label_lower_bound', y_lower_bound[train_index])
    dtrain.set_float_info('label_upper_bound', y_upper_bound[train_index])
    dvalid = xgb.DMatrix(X.values[valid_index, :])
    dvalid.set_float_info('label_lower_bound', y_lower_bound[valid_index])
    dvalid.set_float_info('label_upper_bound', y_upper_bound[valid_index])
    dtest = xgb.DMatrix(X_test.values)
    dtest.set_float_info('label_lower_bound', test_y_lower_bound)
    dtest.set_float_info('label_upper_bound', test_y_upper_bound)

    params = {'verbosity': 0,
              'objective': 'survival:aft',
              'eval_metric': 'aft-nloglik',
              'tree_method': 'hist',
              'learning_rate': 0.1,
              'aft_loss_distribution': 'normal',
              'aft_loss_distribution_scale': 1.20,
              'max_depth': 6,
              'lambda': 0.1,
              'alpha': 0.05}
    bst = xgb.train(params, dtrain, num_boost_round=10000,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=50)
    print('模型参数：', params)
    df = pd.DataFrame({'Label (lower bound)': test_y_lower_bound,
                       'Label (upper bound)': test_y_upper_bound,
                       'Predicted label': bst.predict(dtest)})
    rights = [0.0, 0.0, 0.0, 0.0]
    limits = [3, 5, 7, 10]
    classfilter_matrix_1 = [[0.0, 0.0], [0.0, 0.0]]
    classfilter_matrix_2 = [[0.0, 0.0], [0.0, 0.0]]
    for (x, y) in zip(df['Label (lower bound)'], df['Predicted label']):
        for i in range(len(limits)):
            rights[i] += 1 if abs(x - y) <= limits[i] else 0
        classfilter_matrix_1[0][0] += 1 if x <= 15 and y <= 15 else 0
        classfilter_matrix_1[0][1] += 1 if x <= 15 and y > 15 else 0
        classfilter_matrix_1[1][0] += 1 if x > 15 and y <= 15 else 0
        classfilter_matrix_1[1][1] += 1 if x > 15 and y > 15 else 0

        classfilter_matrix_2[0][0] += 1 if x <= 30 and y <= 30 else 0
        classfilter_matrix_2[0][1] += 1 if x <= 30 and y > 30 else 0
        classfilter_matrix_2[1][0] += 1 if x > 30 and y <= 30 else 0
        classfilter_matrix_2[1][1] += 1 if x > 30 and y > 30 else 0

    for i in range(len(limits)):
        print('<={}'.format(limits[i]), rights[i] / len(df), rights[i], len(df))

    print('<=15准确率:', classfilter_matrix_1[0][0] / (classfilter_matrix_1[0][0] + classfilter_matrix_1[1][0]))
    print('>15准确率:', classfilter_matrix_1[1][1] / (classfilter_matrix_1[0][1] + classfilter_matrix_1[1][1]))
    print('<=15召回率:', classfilter_matrix_1[0][0] / (classfilter_matrix_1[0][0] + classfilter_matrix_1[0][1]))
    print('>15召回率:', classfilter_matrix_1[1][1] / (classfilter_matrix_1[1][0] + classfilter_matrix_1[1][1]))

    print('<=30准确率:', classfilter_matrix_2[0][0] / (classfilter_matrix_2[0][0] + classfilter_matrix_2[1][0]))
    print('>30准确率:', classfilter_matrix_2[1][1] / (classfilter_matrix_2[0][1] + classfilter_matrix_2[1][1]))
    print('<=30召回率:', classfilter_matrix_2[0][0] / (classfilter_matrix_2[0][0] + classfilter_matrix_2[0][1]))
    print('>30召回率:', classfilter_matrix_2[1][1] / (classfilter_matrix_2[1][0] + classfilter_matrix_2[1][1]))


def xgb_light_cv(data_train, data_test):
    # data,lable 分开
    data_train.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)
    label_train = data_train.pop('typing_length')
    label_test = data_test.pop('typing_length')
    model = XGBClassifier(learning_rate=0.2,
                          n_estimators=200,
                          max_depth=8,
                          min_child_weight=1,
                          gamma=0.1,
                          reg_lambda=0,
                          reg_alpha=0,
                          subsample=0.9,
                          colsample_bytree=0.9,
                          objective='binary:logitraw',
                          eval_metric='logloss',
                          early_stopping_rounds=20,
                          scale_pos_weight=1,
                          )
    param_test = {'max_depth': np.linspace(5, 10, 6, dtype=int),
                  'min_child_weight': np.linspace(1, 10, 5, dtype=int),
                  'gamma': np.linspace(0, 0.5, 6),
                  'learning_rate': np.logspace(-2, 0, 10)
                  }
    gsearch = GridSearchCV(model, param_grid=param_test, verbose=2, cv=3, n_jobs=-1)
    gsearch.fit(data_train, label_train)
    print(gsearch.best_params_, gsearch.best_score_)


def xgb_random_cv(data_train, data_test):
    # data,lable 分开
    data_train.reset_index(inplace=True, drop=True)
    data_test.reset_index(inplace=True, drop=True)
    label_train = data_train.pop('typing_length')
    label_test = data_test.pop('typing_length')

    ndcg_score = make_scorer(ndcg)
    parameters = {
        'max_depth': [4, 6, 8, 10],
        'learn_rate': [0.01, 0.02],
        'n_estimators': [100, 300],
        'min_child_weight': [0, 2, 5, 10],
        'subsample': [0.7, 0.8, 0.85, 0.95],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    model = xgb.sklearn.XGBClassifier(
        nthread=5,
        learn_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='rank:pairwise',
        n_estimators=10
    )
    gsearch = RandomizedSearchCV(model, param_distributions=parameters, scoring=ndcg_score, cv=3)
    print('gridsearchcv fit begin...')
    gsearch.fit(data_test, label_test)
    print('Best score: {}'.format(gsearch.best_score_))
    print('Best parameters set: {}'.format(gsearch.best_estimator_.get_params()))


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def ndcg(y_true, y_pred, k=20, rel_threshold=0):
    if k <= 0:
        return 0
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c_g = sorted(c, key=lambda x: x[0], reverse=True)
    c_p = sorted(c, key=lambda x: x[1], reverse=True)
    idcg = 0
    ndcg = 0
    for i, (g, p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            idcg += (math.pow(2, g) - 1) / math.log(2 + i)
    for i, (g, p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            ndcg += (math.pow(2, g) - 1) / math.log(2 + i)
    if idcg == 0:
        return 0
    else:
        return ndcg / idcg


def model_valid1(y_pred, y_test):
    total = len(y_test)
    rights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sort_y_pred = y_pred.sort_values().index.tolist()
    sort_y_test = y_test.sort_values().index.tolist()
    for i in range(len(sort_y_test)):
        j = sort_y_pred.index(sort_y_test[i])
        if abs(i - j) <= total * 0:
            rights[0] += 1
        if abs(i - j) <= total * 0.1:
            rights[1] += 1
        if abs(i - j) <= total * 0.2:
            rights[2] += 1
        if abs(i - j) <= total * 0.3:
            rights[3] += 1
        if abs(i - j) <= total * 0.4:
            rights[4] += 1
        if abs(i - j) <= total * 0.5:
            rights[5] += 1
    for right in rights:
        print(right, total, right / float(total))


def model_valid2(y_pred, y_test):
    total = len(y_test) * (len(y_test) - 1) / 2
    right = 0
    for i in range(len(y_test) - 1):
        label1 = y_test[i]
        pred1 = y_pred[i]
        for j in range(i + 1, len(y_test)):
            label2 = y_test[j]
            pred2 = y_pred[j]
            if (label1 - label2) * (pred1 - pred2) > 0:
                right += 1
    print(right, total, right / float(total))


def evalerror(y_pred, dtrain):
    diff = abs(y_pred - dtrain.get_label())
    right = 0.0
    for num in diff:
        right += 1 if num <= 1 else 0
    return 'error', 1 - right / float(len(y_pred))


def model_valid3(y_pred, y_test):
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    diff = abs(y_pred - y_test)

    for i in list(range(0, 10, 2)):
        right = diff.apply(lambda x: 1 if x <= i else 0).sum()
        print('<={}'.format(i), right / float(len(diff)), "({} / {})".format(right, float(len(diff))))


def model_valid3_1(y_pred, y_test):
    y_pred.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    totals = [0.0, 0.0, 0.0, 0.0]
    totals_1 = [0.0, 0.0, 0.0, 0.0]
    rights = [0.0, 0.0, 0.0, 0.0]
    diff = [0, 15, 30, 60]
    for i in range(len(y_pred)):
        for j in range(len(diff) - 1):
            rights[j] += 1 if diff[j] < y_pred[i] <= diff[j + 1] and diff[j] < y_test[i] <= diff[j + 1] else 0
            totals[j] += 1 if diff[j] < y_test[i] <= diff[j + 1] else 0
            totals_1[j] += 1 if diff[j] < y_pred[i] <= diff[j + 1] else 0
    for i in range(len(diff) - 1):
        print(
            '{} ~ {}准确率: {} ({} / {})'.format(diff[i], diff[i + 1], rights[i] / totals_1[i] if totals_1[i] != 0 else 0,
                                              rights[i], totals_1[i]))
        print('{} ~ {}召回率: {} ({} / {})'.format(diff[i], diff[i + 1], rights[i] / totals[i] if totals[i] != 0 else 0,
                                                rights[i], totals[i]))


def model_valid3_2(y_pred, y_test):
    sums = np.zeros(shape=[1, 3]).reshape(-1)
    righs = np.zeros(shape=[1, 3]).reshape(-1)
    for (pred, test) in zip(y_pred.values.tolist(), y_test.values.tolist()):
        if pred in [0, 1, 2] and test in [0, 1, 2]:
            righs[0] += 1.0
        elif pred in [3, 4, 5] and test in [3, 4, 5]:
            righs[1] += 1.0
        elif pred in [6, 7, 8, 9] and test in [6, 7, 8, 9]:
            righs[2] += 1.0

        if pred in [0, 1, 2]:
            sums[0] += 1.0
        elif pred in [3, 4, 5]:
            sums[1] += 1.0
        elif pred in [6, 7, 8, 9]:
            sums[2] += 1.0
    for i in range(3):
        print(i, righs[i] / sums[i], '({} / {})'.format(righs[i], sums[i]))
    print(sum(righs) / len(y_pred))


def model_valid4(model, data_train, y_test):
    print('类别 0:[0,{}), 1:[{}, )'.format(div_num, div_num))
    print('数量 0:{}, 1:{}'.format(y_test.apply(lambda x: 1 if x < 1 else 0).sum(),
                                 y_test.apply(lambda x: 1 if x >= 1 else 0).sum()))
    y_pred = pd.Series(model.predict(data_train))
    # print '类别 0:[0,10), 1:[10,30), 2:[30,)'
    # print '类别 0:[0,7), 1:[7,15), 2:[15,30), 3:[30,)'
    # print '样本数 0:{}, 1:{}, 2:{}, 3:{}'.format(y_test.apply(lambda x: 1 if x == 0 else 0).sum(), y_test.apply(lambda x: 1 if x == 1 else 0).sum(), y_test.apply(lambda x: 1 if x == 2 else 0).sum(), y_test.apply(lambda x: 1 if x == 3 else 0).sum())
    print('准确率：', accuracy_score(y_test, y_pred),
          '({} / {})'.format(accuracy_score(y_test, y_pred, normalize=False), len(y_pred)))
    print('查准率：', precision_score(y_test, y_pred, average='weighted'),
          '(  / {})'.format(y_pred.apply(lambda x: 1 if x >= 1 else 0).sum()))
    print('召回率：', recall_score(y_test, y_pred, average='weighted'),
          '(  / {})'.format(y_test.apply(lambda x: 1 if x >= 1 else 0).sum()))
    print(classification_report(y_test, y_pred, target_names=['0', '1']))


def model_valid5(model, data_train, y_test):
    data_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    y_pred = model.predict_proba(data_train)
    total, right = 0, 0
    pred_totals = [0, 0, 0, 0]
    test_totals = [0, 0, 0, 0]
    rights = [0, 0, 0, 0]
    for i in range(len(y_test)):
        test_totals[y_test[i]] += 1
        for j in range(2):
            if y_pred[i][j] >= 0.7:
                total += 1
                pred_totals[j] += 1
                if y_test[i] == j:
                    rights[j] += 1
                    right += 1

    print('>=0.7')
    print('准确率:', right / float(total), '({} / {})'.format(right, total))
    print('召回率:', right / float(len(y_test)), '({} / {})'.format(right, len(y_test)))
    print('---------')
    for i in range(2):
        print('{} 准确率:'.format(i), 0 if pred_totals[i] == 0 else rights[i] / float(pred_totals[i]),
              '({} / {})'.format(rights[i], pred_totals[i]))
    print('---------')
    for i in range(3):
        print('{} 召回率:'.format(i), 0 if test_totals[i] == 0 else rights[i] / float(test_totals[i]),
              '({} / {})'.format(rights[i], test_totals[i]))


def save_csv(pd_data, pred, label):
    res = pd_data
    res['pred0'] = pred
    # res['pred1'] = pred[:, 1]
    res['label'] = label
    res['diff'] = abs(pred - label)
    res.to_csv('pred.csv', index=False)


if __name__ == '__main__':
    data_train, data_test = load_data()
    data_train, data_test = clean_data_v2(data_train, data_test)
    # data_train, data_test = cut_data(data_train, data_test)
    # data_train, data_test = filter_near_data(data_train, data_test)
    predict(data_train, data_test)
