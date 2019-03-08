# -*- coding: utf-8 -*-
"""
Package features in step3.2 and step3.3 and save as numpy array

@author: Marmot
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

data_folder = '../../data/'

set_name_list = ['train', 'testAB']

for set_name in set_name_list:
    input_file = data_folder + set_name + '_sample_data'
    input_size = pd.read_csv(data_folder + set_name + '_sample_size.csv')
    if (set_name == 'train'):
        pic_sample = pd.read_csv(data_folder + set_name + '_pic_sample.csv')
        train_label = pd.read_csv(data_folder + 'train_label.csv', names=['value'])
        train_label['PIC_IND'] = train_label.index + 1
        pic_sample = pd.merge(pic_sample, train_label, how='left', on='PIC_IND')
    if set_name == 'testAB':
        pic_sample = pd.read_csv(data_folder + set_name + 'B_pic_sample.csv')

    F3 = pd.read_csv(data_folder + set_name + '_F3.csv')
    pic_sample = pd.merge(pic_sample, F3, on='PIC_IND', how='left')

    F2 = pd.read_csv(data_folder + set_name + '_F2.csv')
    pic_sample = pd.merge(pic_sample, F2, on='PIC_IND', how='left')

    # %% general descirption
    velo_ = ['V' + str(x).zfill(2) for x in np.arange(1, 7)]
    coord_ = ['C' + str(x).zfill(2) for x in np.arange(1, 6)]
    N_centroids = 8
    kp_ = ['kp' + str(x).zfill(2) for x in np.arange(1, N_centroids + 1)]
    hist_ = ['H' + str(x).zfill(2) for x in np.arange(1, 4)]
    bin_ = ['B' + str(x).zfill(2) for x in np.arange(1, 8)]
    M_ = ['M' + str(x).zfill(2) for x in np.arange(1, 4)]
    R_ = ['R' + str(x).zfill(2) for x in np.arange(1, 3)]
    N_ = ['N' + str(x).zfill(2) for x in np.arange(1, 10)]
    #    NP_ = map(lambda x:'NP'+ str(x).zfill(2), np.arange(2,20 ))

    GS_ = velo_ + coord_ + hist_ + bin_ + M_ + R_ + N_
    # %% time space description
    time_diff_list = np.asarray([2, 4])
    cover_diff_ = ['COV_DIFF' + str(x).zfill(2) for x in time_diff_list]
    mean_diff_ = ['MEA_DIFF' + str(x).zfill(2) for x in time_diff_list]
    std_diff_ = ['STD_DIFF' + str(x).zfill(2) for x in time_diff_list]
    max_diff_ = ['MAX_DIFF' + str(x).zfill(2) for x in time_diff_list]
    height_diff_list = np.asarray([3, 4])
    cover_diff_H = ['COV_DIFF_H' + str(x).zfill(2) for x in height_diff_list]
    mean_diff_H = ['MEA_DIFF_H' + str(x).zfill(2) for x in height_diff_list]
    std_diff_H = ['STD_DIFF_H' + str(x).zfill(2) for x in height_diff_list]
    max_diff_H = ['MAX_DIFF_H' + str(x).zfill(2) for x in height_diff_list]
    time_list = np.asarray([11, 15])
    cover_ = ['COV' + str(x).zfill(2) for x in time_list]
    mean_ = ['MEA' + str(x).zfill(2) for x in time_list]
    std_ = ['STD' + str(x).zfill(2) for x in time_list]
    max_ = ['MAX' + str(x).zfill(2) for x in time_list]

    TS_ = cover_diff_ + mean_diff_ + std_diff_ + max_diff_ + cover_diff_H + mean_diff_H + std_diff_H + max_diff_H + cover_ + mean_ + std_ + max_
    TS_ = cover_diff_ + max_diff_ + cover_diff_H + max_diff_H + cover_ + mean_ + max_

    feature_list = velo_ + coord_ + hist_ + bin_ + M_ + R_ + N_
    feature_list = GS_ + TS_

    feature_list_pd = pd.read_csv(data_folder + set_name + '_image_PICIND.csv')
    pic_sample = pic_sample[pic_sample.PIC_IND.isin(feature_list_pd.PIC_IND)]

    X_ = pic_sample[feature_list].values
    if set_name == 'train':
        scaler = preprocessing.MinMaxScaler().fit(X_)
        X = scaler.transform(X_)
    if set_name == 'testAB':
        X = scaler.transform(X_)
    np.save(data_folder + set_name + '_flat.npy', X)
