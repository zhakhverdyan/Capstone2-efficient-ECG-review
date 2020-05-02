#!/usr/bin/env mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:35:49 2020

@author: zhannahakhverdyan
"""

import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

non_beat = ['+', '~', '!', '"', 'x', '|', '[', ']']
path = '../../mit-bih-arrhythmia-database-1.0.0/'
file_list = list(range(100, 235))
list_to_remove = list(range(125,200))
list_to_remove += [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]
for item in list_to_remove:
    file_list.remove(item)
file_list = [str(item) for item in file_list]


def make_data_1beat_per_row(path, file_name, window): # window in sec correspond to 360 points of the signal
    """Chop  the ecg signal to 1 sec intervals with beat annotation at the center"""
    signals, fields = wfdb.rdsamp(path+file_name, channels=[0]) # get the signal for upper channel
    annotation = wfdb.rdann(path+file_name, 'atr', return_label_elements=['symbol']) # get annotations
    samp_index = annotation.sample.tolist() # make a list of annotation indices
    annot_symbol = annotation.symbol # make a list of annotation symbols
    step = int(window * 360/2) # calculate the sample size on either side of the beat
    x = np.zeros((len(annot_symbol)-12, 2*step)) # empty array, n_cols = window*360, n_rows = number of annotations
    signal_total = np.squeeze(signals).tolist() # get the one-dimmensional signal as list
    for i in np.arange(6, len(samp_index)-6):
        x[i-6, :] = signal_total[samp_index[i]-step:samp_index[i]+step] 
        # get the signal corresponding to the beat index +/- step
    labeled_beat_df = pd.DataFrame(x) # turn the matrix into a DataFrame
    label_column = pd.Series(annot_symbol[6:-6], name='Label')
    labeled_beat_df = pd.concat([labeled_beat_df, label_column], axis=1) # add beat annotations
    
    # generate an additional column for 5 label classes
    labeled_beat_df.loc[labeled_beat_df.Label.isin(['N', 'L', 'R', 'e', 'j']), 'Label_class'] = 'N'
    labeled_beat_df.loc[labeled_beat_df.Label.isin(['A', 'a', 'J', 'S']), 'Label_class'] = 'S'
    labeled_beat_df.loc[labeled_beat_df.Label.isin(['V', 'E']), 'Label_class'] = 'V'
    labeled_beat_df.loc[labeled_beat_df.Label.isin(['F']), 'Label_class'] = 'F'
    labeled_beat_df.loc[labeled_beat_df.Label.isin(['/', 'f', 'Q']), 'Label_class'] = 'Q'
   
    # generate an additional column for numerical output labels
    labeled_beat_df.loc[labeled_beat_df.Label_class=='N', 'Output_label'] = 0
    labeled_beat_df.loc[labeled_beat_df.Label_class=='S', 'Output_label'] = 1
    labeled_beat_df.loc[labeled_beat_df.Label_class=='V', 'Output_label'] = 2
    labeled_beat_df.loc[labeled_beat_df.Label_class=='F', 'Output_label'] = 3
    labeled_beat_df.loc[labeled_beat_df.Label_class=='Q', 'Output_label'] = 4

    # scale all values in a row between 0 and 1, to do this need to transpose the matrix first
    scaler = MinMaxScaler()
    x = labeled_beat_df.drop(columns=['Label', 'Label_class', 'Output_label'])
    x_scaled = scaler.fit_transform(x.T).T
    df_scaled = pd.DataFrame(x_scaled)
    df_scaled = pd.concat([df_scaled, labeled_beat_df[['Label', 'Label_class', 'Output_label']]], axis=1)

    # filter out non-beat signals
    df_scaled = df_scaled[~df_scaled.Label.isin(non_beat)]

    # drop any missing values
    df_scaled.dropna(inplace=True)

    return df_scaled


def make_dataset_5class(list_file_names, window): # window in sec correspond to 360 points of the signal
    
    dataset_df = pd.DataFrame() #initialize an empty datafeame

    for file_name in list_file_names:
        signals, fields = wfdb.rdsamp(path+file_name, channels=[0]) # get the signal for upper channel
        annotation = wfdb.rdann(path+file_name, 'atr', return_label_elements=['symbol']) # get annotations
        samp_index = annotation.sample.tolist() # make a list of annotation indices
        annot_symbol = annotation.symbol # make a list of annotation symbols
        step = int(window * 360/2) # calculate the sample size on either side of the beat
        num_rows = len(annot_symbol)-12
        x = np.zeros((num_rows, 2*step)) # empty array, n_cols = window*360, n_rows = number of annotations
        signal_total = np.squeeze(signals).tolist() # get the one-dimmensional signal as list
        for i in np.arange(6, len(samp_index)-6):
            x[i-6, : ] = signal_total[samp_index[i]-step:samp_index[i]+step]
            # get the signal corresponding to the beat index +/- step
        labeled_beat_df = pd.DataFrame(x) # turn the matrix into a DataFrame
        patient_num_column = pd.Series(num_rows*[int(file_name)], name='Patient_number')
        label_column = pd.Series(annot_symbol[6:-6], name='Label')
        labeled_beat_df = pd.concat([labeled_beat_df, patient_num_column, label_column], axis=1) # add patient number and beat annotations
        dataset_df = pd.concat([dataset_df, labeled_beat_df], axis=0, ignore_index=True) # combine with the dataset
    
    # generate an additional column for 5 label classes
    dataset_df.loc[dataset_df.Label.isin(['N', 'L', 'R', 'e', 'j']), 'Label_class'] = 'N'
    dataset_df.loc[dataset_df.Label.isin(['A', 'a', 'J', 'S']), 'Label_class'] = 'S'
    dataset_df.loc[dataset_df.Label.isin(['V', 'E']), 'Label_class'] = 'V'
    dataset_df.loc[dataset_df.Label.isin(['F']), 'Label_class'] = 'F'
    dataset_df.loc[dataset_df.Label.isin(['/', 'f', 'Q']), 'Label_class'] = 'Q'
   
    # generate an additional column for numerical output labels
    dataset_df.loc[dataset_df.Label_class=='N', 'Output_label'] = 0
    dataset_df.loc[dataset_df.Label_class=='S', 'Output_label'] = 1
    dataset_df.loc[dataset_df.Label_class=='V', 'Output_label'] = 2
    dataset_df.loc[dataset_df.Label_class=='F', 'Output_label'] = 3
    dataset_df.loc[dataset_df.Label_class=='Q', 'Output_label'] = 4
    
    # filter out any missing values
    dataset_df = dataset_df[~dataset_df['Label'].isna()]

    # scale all values in a row between 0 and 1, to do this need to transpose the matrix first
    scaler = MinMaxScaler()
    x = dataset_df.drop(columns=['Patient_number', 'Label', 'Label_class', 'Output_label'])
    x_scaled = scaler.fit_transform(x.T).T
    df_scaled = pd.DataFrame(x_scaled)
    df_scaled = pd.concat([df_scaled, dataset_df[['Patient_number', 'Label', 'Label_class', 'Output_label']]], axis=1)

    # filter out non-beat signals
    df_scaled = df_scaled[~df_scaled.Label.isin(non_beat)]

    # drop any missing values
    df_scaled.dropna(inplace=True)

    return df_scaled

def plot_ave_std(dataset, dataset_name):
    """Function takes a dataset and name, e.g. training set and plots 
    5 subplots, 1 for each beat class, with average +/-1 std"""
    data_cols = dataset.drop('Output_label', axis=1).columns.tolist()
    ave = dataset.groupby('Output_label')[data_cols].mean().sort_index()
    stdev = dataset.groupby('Output_label')[data_cols].std().sort_index()
    ave_stdev = stdev.mean(axis=1)
    count = dataset.groupby('Output_label')[data_cols].count().sort_index()
    num_subplots = len(ave)
    fig,ax =  plt.subplots(1,num_subplots)
    fig.set_size_inches(num_subplots*3,3)
    fig.subplots_adjust(top=0.7)
    beat_titles = ['(N), ', '(S), ', '(V), ', '(F), ', '(Q), ']
    if num_subplots>1:
        for i in range(num_subplots):
            ax[i].plot(np.arange(0,360), ave.iloc[i,:])
            lower_bound = ave.iloc[i,:]-stdev.iloc[i,:]
            upper_bound = ave.iloc[i,:]+stdev.iloc[i,:]
            ax[i].fill_between(np.arange(0,360), lower_bound, upper_bound, facecolor='cornflowerblue', alpha=0.25)
            title = beat_titles[int(ave.index.tolist()[i])] + str(count.values[i,0]) + ', ' +str(round(ave_stdev.values[i],3))
            ax[i].set_title(title)
            #label = ["ave stdev {}".format(round(ave_stdev.values[i],3))]
            #ax[i].legend(label, loc="upper right")
    else:
        ax.plot(np.arange(0,360), ave.iloc[0,:])
        lower_bound = ave-stdev
        upper_bound = ave+stdev
        ax.fill_between(np.arange(0,360), lower_bound.iloc[0,:], upper_bound.iloc[0,:], facecolor='cornflowerblue', alpha=0.25)
        title = beat_titles[int(ave.index.tolist()[0])] + str(count.values[0,0]) + ', ' +str(round(ave_stdev.values[0],3))
        ax.set_title(title)
        #label = ["ave stdev {}".format(round(ave_stdev.values[0],3))]
        #ax.legend(label, loc="upper right")
    title = "Average and 1 sigma range in the "+dataset_name+'.\nClass, count, average sigma for each class'    
    fig.suptitle(title)
    plt.show()