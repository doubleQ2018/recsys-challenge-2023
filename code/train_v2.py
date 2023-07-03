# -*- coding:utf-8 -*-
#========================================================================
# Copyright (C) 2022 All rights reserved.
# Author: doubleQ
# File Name: train.py
# Created Date: 2022-07-23
# Description:
# =======================================================================

import datetime
import gc
import io
import os
import sys
import base64
import glob
import itertools
import numpy as np
import pandas as pd
import random
import sklearn
import pickle
from itertools import combinations
from functools import reduce
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import multiprocessing

from lightgbm import early_stopping
import lightgbm as lgb
from lightgbm.sklearn import LGBMRanker
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

SEED = 2023
FOLD = 15
version = 'v2'
model_name = 'xgb'
mode = 'online'

def cv_model(model_name, train_x, train_y, test_x, cat_features=[], prefix='', groups=None):

    if model_name == 'lgb':
        clf = lgb
    elif model_name == 'xgb':
        clf = xgb
    else:
        clf = CatBoostClassifier

    folds = FOLD
    seed = SEED
    if groups is None:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_split = kf.split(train_x)
    else:
        kf = GroupKFold(n_splits=folds)
        fold_split = kf.split(train_x, groups=groups)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])
    models = dict()
    importance_df = pd.DataFrame()
    for i, (train_index, valid_index) in enumerate(fold_split):
        print('***********************************{} fold {} ************************************'.format(prefix, str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        os.makedirs('models', exist_ok=True)
        model_save = 'models/{}_{}_fold{}.pkl'.format(prefix, model_name, i+1)

        if model_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                #'metric': 'auc',
                'metric': 'binary_logloss',
                'max_depth': 8,
                'num_leaves': 2 ** 6,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,  
                'bagging_fraction': 0.8, 
                'bagging_freq': 5,  
                'learning_rate': 0.01,  
                'verbose': -1,
                'device': 'cpu',
                'feature_fraction_seed':SEED,
                'bagging_seed':SEED,
                'seed': SEED,
                'n_jobs': 8
            }

            if not os.path.exists(model_save):
                model = clf.train(params, train_matrix, 10000, valid_sets=[train_matrix, valid_matrix],
                              categorical_feature=cat_features, verbose_eval=1000, early_stopping_rounds=100)
                save_file(model_save, model)
            else:
                model = load_file(model_save)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})
            importances.sort_values('importances',ascending=False,inplace=True)
            print(importances.iloc[:20])

        if model_name == "xgb":
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)
            test_matrix = clf.DMatrix(test_x)

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      #'eval_metric': 'auc',
                      'eval_metric': 'logloss',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'learning_rate': 0.01,
                      'max_depth': 10,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 0.8,
                      'eta': 0.1,
                      'gpu_id': 0,
                      'tree_method': 'gpu_hist',
                      'seed': SEED,
                      'nthread': 8,
                      }

            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]

            if not os.path.exists(model_save):
                model = clf.train(params, train_matrix, num_boost_round=10000, evals=watchlist, verbose_eval=1000, early_stopping_rounds=500)
                save_file(model_save, model)
            else:
                model = load_file(model_save)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)

            importances = pd.DataFrame(list(model.get_fscore().items()),
                columns=['features', 'importances']).sort_values('importances', ascending=False)
            importances.sort_values('importances',ascending=False,inplace=True)
            print(importances.iloc[:20])

        if model_name == "cat":
            params = {
                    'n_estimators': 50000,
                    'learning_rate': 0.01,
                    'eval_metric':'Logloss',
                    'loss_function':'Logloss',
                    'random_seed':SEED,
                    'metric_period':5000,
                    'one_hot_max_size': 254,
                    'od_wait':500,
                    'depth': 8,
                    'task_type': 'GPU',
                    'gpu_ram_part': 0.6,
                    'verbose': True,
                    }
            model = clf(**params)
            if not os.path.exists(model_save):
                model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=cat_features, use_best_model=True)
                save_file(model_save, model)
            else:
                model = load_file(model_save)
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            importances = pd.DataFrame({'features':train_x.columns,
                                    'importances':model.feature_importances_})
            importances.sort_values('importances',ascending=False,inplace=True)
            #print(importances.iloc[:20])
        models['{}_{}'.format(model_name, i+1)] = model
        importance_df = pd.concat([importance_df, importances], axis=0)
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits

    return train, test, models, importance_df

def save_file(file_path, model):
    with open(file_path, 'wb') as fout:
        pickle.dump(model, fout, protocol=4)

def load_file(file_path):
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)

def predict(model_name, test_x, prefix=''):
    if model_name == 'lgb':
        clf = lgb
    elif model_name == 'xgb':      
        clf = xgb
    else:
        clf = CatBoostClassifier
    test = np.zeros(test_x.shape[0])
    for i in range(FOLD):
        model_save = '/home/mw/project/save_models/{}_{}_fold{}.txt'.format(prefix, model_name, i+1)
        if model_name == 'lgb':
            model = clf.Booster(model_file=model_save)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        elif model_name == 'xgb':
            test_matrix = clf.DMatrix(test_x)
            model = pickle.load(open(model_save, 'rb'))
            test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
        else:
            model = clf()
            model.load_model(model_save)
            test_pred = model.predict_proba(test_x)[:,1]
        test += test_pred / FOLD
    return test

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def cross_feature(data, keys=[]):
    features = keys
    feats = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            le = LabelEncoder()
            data[f'{features[i]}+{features[j]}'] = data[features[i]].astype(str) + data[features[j]].astype(str)
            feats.append(f'{features[i]}+{features[j]}')
    return data, feats

def static(df, key, value_dict, prefix=""):
    if not isinstance(key, list):
        key = [key]
    df = df.groupby(key).aggregate(value_dict)
    df.columns = ["{}{}_{}_{}".format(prefix, "+".join(key), k, vv if isinstance(vv, str) else vv.__name) for k, v in value_dict.items() for vv in (v if isinstance(v, list) else [v])]
    return df.reset_index()

def build_ctr(data, keys):
    dfs = None
    end_time = data.f_1.max()
    for day in range(data.f_1.min(), data.f_1.max()+1):
        df = data[data.f_1 == day]
        for offset in range(1, 8):
            history = data[(data.f_1 >= day-offset) & (data.f_1 < day)]
            for k in keys:
                ctr = static(history, k, {'is_clicked': ['mean'], 'is_installed': ['mean']}, prefix='offset_{}_'.format(offset))
                df = df.merge(ctr, on=k, how='left')
        for k in keys:
            value_dict = {f: ['count', 'nunique'] for f in keys if f != k}
            sta = static(df, k, value_dict, prefix='static_')
            df = df.merge(sta, on=k, how='left')
            value_dict = {f: ['mean', 'max', 'min', 'std', 'sum'] for f in ['f_42', 'f_57', 'f_55', 'f_61']}
            sta = static(df, k, value_dict, prefix='static_')
            df = df.merge(sta, on=k, how='left')
        dfs = pd.concat([dfs, df], ignore_index=True)
    data = dfs.sort_values(by='f_1')
    return data

DATA_PATH = '../sharechat_recsys2023_data'
sparse_feat = ['f_{}'.format(i) for i in range(2, 42)]
dense_feat = ['f_{}'.format(i) for i in range(42, 80)]

# train test data preprocess
data_path = '{}/data_{}_{}.pkl'.format(DATA_PATH, mode, version)
if os.path.exists(data_path):
    train_data, test_data, sparse_feat = load_file(data_path)
else:
    fs = []
    for f in glob.glob(os.path.join(DATA_PATH, 'train', '*.csv')):
        fs.append(pd.read_csv(f, sep='\t'))
    train_df = pd.concat(fs, ignore_index=False)
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test', '000000000000.csv'), sep='\t')

    data = pd.concat([train_df, test_df], ignore_index=False)

    data[sparse_feat] = data[sparse_feat].fillna(-1)
    data[dense_feat] = data[dense_feat].fillna(0)
    data['user_id'] = data[['f_{}'.format(i) for i in range(43, 55)]].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
    data['item_id'] = data[['f_{}'.format(i) for i in range(64, 71)]].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
    sparse_feat += ['user_id', 'item_id']
    data, cross_feat = cross_feature(data, keys=['user_id', 'item_id', 'f_4', 'f_11', 'f_6', 'f_15', 'f_17', 'f_12', 'f_13', 'f_18'])
    sparse_feat += cross_feat

    for feat in sparse_feat:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])

    data = build_ctr(data, keys=['user_id', 'item_id', 'f_4', 'f_11', 'f_6', 'f_15', 'f_17', 'f_12', 'f_13', 'f_18'])
    # 19days train data
    if mode == 'offline':
        train_data = data[(data.f_1 > 46) & (data.f_1 <= 65)].reset_index(drop=True)
        test_data = data[data.f_1 == 66].reset_index(drop=True)
    else:
        train_data = data[(data.f_1 > 47) & (data.f_1 <= 66)].reset_index(drop=True)
        test_data = data[data.f_1 == 67].reset_index(drop=True)
    train_data = reduce_mem_usage(train_data)
    test_data = reduce_mem_usage(test_data)
    save_file(data_path, (train_data, test_data, sparse_feat))
print(list(train_data.columns))
print(train_data.shape)
print(test_data.shape)

drop_list = ['is_clicked', 'is_installed'] + ['f_0', 'f_1']

result = test_data[['f_0']]
for target in ['is_installed']:
    x_train = train_data.drop(drop_list, axis=1 )
    y_train = train_data[target]
    x_test = test_data.drop( drop_list, axis=1 )
    train, test, model, feature_importance_df = cv_model(model_name, x_train, y_train, x_test, cat_features=[], prefix='seed'+str(SEED)+'_'+version+'_'+target, groups=None)
    all_features = feature_importance_df[['features', 'importances']].groupby('features').mean().sort_values(
            by='importances', ascending=False
        ).reset_index()
    print(all_features.iloc[:50])
    train_auc = roc_auc_score(y_train, train)
    train_logloss = log_loss(y_train, train, eps=1e-7, labels=[0,1], normalize=True)
    print('{}, train auc: {:.6f}, train logloss: {:.4f}'.format(target, train_auc, train_logloss))
    result[target] = test

if mode == 'offline':
    for target in ['is_installed']:
        valid_auc = roc_auc_score(test_data[target], result[target])
        valid_logloss = log_loss(test_data[target], result[target], eps=1e-7, labels=[0,1], normalize=True)
        print('{}, valid auc: {:.6f}, valid logloss: {:.4f}'.format(target, valid_auc, valid_logloss))
else:
    os.makedirs('result', exist_ok=True)
    result['is_clicked'] = result['is_installed']
    result.columns = ['RowId', 'is_clicked', 'is_installed']
    result.to_csv('result/{}_{}_{}_fold{}_seed{}.csv'.format(model_name, version, train_logloss, FOLD, SEED), index=False, sep='\t')

