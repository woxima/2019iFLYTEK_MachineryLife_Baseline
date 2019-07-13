# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:17:03 2019

@author: 大风君

@榜样：鱼佬
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


# 获取数据文件地址
def getfilelist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            getfilelist(newdir, filelist)
    return filelist

#处理统计数据
def stat(data,c,name):
        c[name + '_max'] = data.max()
        c[name + '_min'] = data.min()
        c[name + '_count'] = data.count()
        c[name + '_mean'] = data.mean()
        c[name + '_ptp'] = data.ptp()
        c[name + '_std'] = data.std()
        return c

#处理单个训练样本
def process_sample_single(e,train_p):
    data = pd.read_csv(e)
    lifemax=data['部件工作时长'].max()
    data=data[data['部件工作时长']<=lifemax*train_p]
    c = {'train_file_name': os.path.basename(e)+str(train_p),
         '开关1_sum':data['开关1信号'].sum(),
         '开关2_sum':data['开关2信号'].sum(),
         '告警1_sum':data['告警信号1'].sum(),
         '设备':data['设备类型'][0],
         'life':lifemax-data['部件工作时长'].max()
         }
    for i in ['部件工作时长', '累积量参数1', '累积量参数2',
              '转速信号1','转速信号2','压力信号1','压力信号2',
              '温度信号','流量信号','电流信号']:
        c=stat(data[i],c,i)
    this_tv_features = pd.DataFrame(c, index=[0])  
    
    return this_tv_features

# 多进程调用单文件处理函数，并整合到一起
def get_together(cpu, listp,istest,func):

    if istest :
            train_p_list=[1]
            rst = []
            pool = Pool(cpu)
            for e in listp:
                for train_p in train_p_list:
                    rst.append(pool.apply_async(func, args=(e,train_p,)))
            pool.close()
            pool.join()
            rst = [i.get() for i in rst]
            tv_features=rst[0]
            for i in rst[1:]:
                tv_features = pd.concat([tv_features, i], axis=0)
            cols=tv_features.columns.tolist()
            for col in [idx,ycol]:
                cols.remove(col)
            cols=[idx]+cols+[ycol]
            tv_features[idx]=tv_features[idx].apply(lambda x:x[:-1])
            tv_features=tv_features.reindex(columns=cols)
    else:   
        train_p_list=[0.45,0.55,0.63,0.75,0.85]
        rst = []
        pool = Pool(cpu)
        for e in listp:
            for train_p in train_p_list:
                rst.append(pool.apply_async(func, args=(e,train_p, )))
        pool.close()
        pool.join()
        rst = [i.get() for i in rst]
        tv_features=rst[0]
        for i in rst[1:]:
            tv_features = pd.concat([tv_features, i], axis=0)
        cols=tv_features.columns.tolist()
        for col in [idx,ycol]:
            cols.remove(col)
        cols=[idx]+cols+[ycol]
        tv_features=tv_features.reindex(columns=cols)

    return tv_features

#评价指标
def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res

#lgb
def lgb_cv(train, params, fit_params,feature_names, nfold, seed,test):
    train_pred = pd.DataFrame({
        'true': train[ycol],
        'pred': np.zeros(len(train))})
    test_pred = pd.DataFrame({idx: test[idx], ycol: np.zeros(len(test))},columns=[idx,ycol])
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train)):
        print(f'\nFold_{fold_id} Training ================================\n')
        lgb_trn = lgb.Dataset(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx][ycol],
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx][ycol],
            feature_name=feature_names)
        lgb_reg = lgb.train(params=params, train_set=lgb_trn, **fit_params,
                  valid_sets=[lgb_trn, lgb_val])
        val_pred = lgb_reg.predict(
            train.iloc[val_idx][feature_names],
            num_iteration=lgb_reg.best_iteration)
        train_pred.loc[val_idx, 'pred'] = val_pred
        test_pred[ycol] += lgb_reg.predict(test[feature_names]) / nfold
    score = compute_loss(train_pred['true'], train_pred['pred'])
    print('\nCV LOSS:', score)
    return test_pred

    

idx='train_file_name'
ycol='life'

# ====== lgb ======
params_lgb = {'num_leaves': 250, 
              'max_depth':5, 
              'learning_rate': 0.02,
              'objective': 'regression', 
              'boosting': 'gbdt',
              'verbosity': -1}

fit_params_lgb = {'num_boost_round': 5000, 
                  'verbose_eval':200,
                  'early_stopping_rounds': 200}

# 执行主进程
if __name__ == '__main__':
    
    start = time.time()
    path = '../data/'
    
    train_list = getfilelist(path + 'train/', [])
    test_list = getfilelist(path + 'test1/', [])
    
    n=4
    func=process_sample_single
    train=get_together(n,train_list,False,func)
    test =get_together(n,test_list,True,func)
    print("done.", start - time.time())

    train_test=pd.concat([train,test],join='outer',axis=0).reset_index(drop=True)
    train_test=pd.get_dummies(train_test,columns=['设备'])
    feature_name=list(filter(lambda x:x not in[idx,ycol],train_test.columns))
    
    sub= lgb_cv(train_test.iloc[:train.shape[0]] ,params_lgb, fit_params_lgb, 
                feature_name, 5,2018,train_test.iloc[train.shape[0]:])

    sub.to_csv('baseline_sub1.csv',index=False)
    print("process(es) done.", start - time.time())
#    
