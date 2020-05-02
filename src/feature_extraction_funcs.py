#!/anaconda3/envs mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:43:54 2020

@author: zhannahakhverdyan

This script contains all the fucntions necessary to extract features from raw ECG 
signal.
"""
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# define global variables
non_beat = ['+', '~', '!', '"', 'x', '|', '[', ']']
path = '../../mit-bih-arrhythmia-database-1.0.0/'
file_list = list(range(100, 235))
list_to_remove = list(range(125,200))
list_to_remove += [102, 104, 107, 110, 120, 204, 206, 211, 216, 217, 218, 224, 225,\
                   226, 227, 229]
for item in list_to_remove:
    file_list.remove(item)
file_list = [str(item) for item in file_list]

# train, val, test split on records
train = [101, 103, 105, 106, 109, 111, 112, 113, 115, 116, 117, 119, 121, 122, 123,\
         200, 201, 202, 205, 207, 208, 209, 212, 219, 221, 222, 230, 231, 232, 233]
    
val = [100, 124, 215, 223]

test = [108, 114, 118, 203, 210, 213, 214, 220, 228, 234]


def dif_next_point(signal):
    """Calculates the difference between consecutive points and returns an array of differences"""
    assert type(signal) == np.ndarray, "Input should be a numpy array"
    from collections import deque
    new_signal = deque(signal)
    new_signal.appendleft(0)
    new_signal.pop()
    difference = [elem1-elem2 for elem1,elem2 in zip(signal, new_signal)] 
    diff_deque = deque(difference)
    diff_deque.popleft()
    return np.array(diff_deque)

def pull_signal_annot(path, file_name):
    """Read the heartbeat signal and associated metadata, return raw and processed signal, symbol
    indices, and symbols"""
    if file_name == '114':
        num_signal = 1
    else:
        num_signal = 0
    signals, fields = wfdb.rdsamp(path+file_name, channels=[num_signal]) # get raw signal and metadata
    assert fields['sig_name'] == ['MLII'], "Signal from lead other than MLII"
    assert fields['fs'] == 360, "frequency other than 360"
    raw_signal = np.squeeze(signals)
    #dif_signal = dif_next_point(raw_signal)
    annotation = wfdb.rdann(path+file_name, 'atr', return_label_elements=['symbol']) # get annotations
    samp_index = annotation.sample.tolist() # make a list of annotation indices
    annot_symbol = annotation.symbol # make a list of annotation symbols
    return raw_signal, samp_index, annot_symbol
    #return raw_signal, dif_signal, samp_index, annot_symbol

def extract_labeled_heartbeat(file_name, signal, samp_index, annot_symbol, window): # window - heartbeat duration
    """Split the signal into 'window' length rows and associate with the annotated symbol"""
    step = int(window * 360/2) # calculate the sample size on either side of the beat
    num_rows = len(annot_symbol)
    x = np.zeros((num_rows, 2*step)) # empty array, n_cols = window*360, n_rows = number of annotations
    length = signal.size
    if samp_index[0]-step<0:
        left_pad = np.zeros(abs(samp_index[0]-step))
        signal = np.append(left_pad, signal) # add the left padded sequence to signal
        samp_index = [index + abs(samp_index[0]-step) for index in samp_index] # offset all the indices by pad
    if samp_index[-1]+step>length:
        right_pad = np.zeros(samp_index[-1]+step-length)
        signal = np.append(signal, right_pad)
    for i in np.arange(len(samp_index)):
        x[i, : ] = signal[samp_index[i]-step:samp_index[i]+step]
        # get the signal corresponding to the beat index +/- step
    labeled_beat_df = pd.DataFrame(x) # turn the matrix into a DataFrame
    patient_num_column = pd.Series(num_rows*[int(file_name)], name='Patient_number')
    label_column = pd.Series(annot_symbol, name='Label')
    labeled_beat_df = pd.concat([labeled_beat_df, patient_num_column, label_column], axis=1) # add patient number and beat annotations
    return labeled_beat_df
    
        
def group_labels(df):
    """Group 15 symbols into 5 classes for classification"""
    assert 'Label' in df.columns, "Labels column is missing"
    # generate an additional column for 5 label classes
    df.loc[df.Label.isin(['N', 'L', 'R', 'e', 'j']), 'Label_class'] = 'N'
    df.loc[df.Label.isin(['A', 'a', 'J', 'S']), 'Label_class'] = 'S'
    df.loc[df.Label.isin(['V', 'E']), 'Label_class'] = 'V'
    df.loc[df.Label.isin(['F']), 'Label_class'] = 'F'
    df.loc[df.Label.isin(['/', 'f', 'Q']), 'Label_class'] = 'Q'
    return df

def code_labels(df):
    """Encode the label classes with numerical symbols"""
    assert 'Label_class' in df.columns, "Grouped label class column is missing"
    df.loc[df.Label_class=='N', 'Output_label'] = 0
    df.loc[df.Label_class=='S', 'Output_label'] = 1
    df.loc[df.Label_class=='V', 'Output_label'] = 2
    df.loc[df.Label_class=='F', 'Output_label'] = 3
    df.loc[df.Label_class=='Q', 'Output_label'] = 4
    return df

def remove_non_beats(df):
    """Filter out rows corresponding to non-beat annotations"""
    df = df[~df.Label.isin(non_beat)]
    allowed_labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
    assert set(df.Label.tolist()).issubset(set(allowed_labels)), "There are illegal labels"
    allowed_groups = ['N', 'S', 'V', 'F', 'Q']
    assert set(df.Label_class.tolist()).issubset(set(allowed_groups)), "Illegal heartbeat groups"
    assert df.isna().sum().sum() == 0 # should not be missing values
    df = df[df['Label_class']!='Q']
    return df
    
def process_one_record(path, file_name, window, signal_type):
    signal, samp_index, annot_symbol =  pull_signal_annot(path, file_name)
    if signal_type == 'raw':
        signal = signal
    elif signal_type == 'raw_abs':
        signal = np.absolute(signal)
    elif signal_type =='dif':
        signal = dif_next_point(signal)
    elif signal_type == 'dif_abs':
        #signal = np.absolute(dif_next_point(signal))
        signal = dif_next_point(np.absolute(signal))
    else:
        print("Wrong signal type")
    labeled_beat_df = extract_labeled_heartbeat(file_name, signal, samp_index, annot_symbol, window)
    labeled_beat_df = group_labels(labeled_beat_df)
    labeled_beat_df = code_labels(labeled_beat_df)
    labeled_beat_df = remove_non_beats(labeled_beat_df)
    return labeled_beat_df
   

def process_whole_dataset(path, file_list, window, signal_type):
    """Master function to produce row and differnce datasets from all files"""
    labeled_beat_full_df = pd.DataFrame()
    for file_name in file_list:
        labeled_beat_df = process_one_record(path, file_name, window, signal_type)
        labeled_beat_full_df = pd.concat([labeled_beat_full_df, labeled_beat_df], axis=0, \
                                             ignore_index=True) # combine recordes in a single dataframe
    return labeled_beat_full_df

def split_train_val_test_random(df, train, val_test):
    """Split the data on train and val_test on records. Next take a random 30% sample
    of val_test as val set and the remainder will be the test set"""
    assert 'Patient_number' in df.columns, "Patient number for records missing"
    train_df = df.loc[df['Patient_number'].isin(train)]
    val_test_df = df.loc[df['Patient_number'].isin(val_test)]
    np.testing.assert_array_equal(train_df.Patient_number.unique(), train)
    np.testing.assert_array_equal(val_test_df.Patient_number.unique(), val_test)
    val_df, test_df = train_test_split(val_test_df, test_size=0.7, stratify=val_test_df.Label_class, random_state=2020)
    return train_df, val_df, test_df


def split_train_val_test(df, train, val, test):
    """Split the dataset into train, val and test based on the assigned file list for each"""
    assert 'Patient_number' in df.columns, "Patient number for records missing"
    train_df = df.loc[df['Patient_number'].isin(train)]
    val_df = df.loc[df['Patient_number'].isin(val)]
    test_df = df.loc[df['Patient_number'].isin(test)]
    np.testing.assert_array_equal(train_df.Patient_number.unique(), train)
    np.testing.assert_array_equal(val_df.Patient_number.unique(), val)
    np.testing.assert_array_equal(test_df.Patient_number.unique(), test)
    #assert train_df.Patient_number.unique() == train
    #assert val_df.Patient_number.unique() == val
    #assert test_df.Patient_number.unique() == test
    return train_df, val_df, test_df

def plot_ave_std(dataset, dataset_name):
    """Function takes a dataset and name, e.g. training set and plots 
    4 subplots, 1 for each beat class, with average +/-1 std"""
    #data_cols = dataset.columns.tolist()[:216]
    data_cols = dataset.drop(columns = ['Patient_number', 'Label', 'Label_class', 'Output_label'], axis=1).columns.tolist()
    ave = dataset.groupby('Output_label')[data_cols].mean().sort_index()
    stdev = dataset.groupby('Output_label')[data_cols].std().sort_index()
    total_stdev = dataset[data_cols].std(axis=0)
    total_stdev_ave = total_stdev.mean()
    #variance = dataset.groupby('Output_label')[data_cols].var().sort_index()
    ave_stdev = stdev.mean(axis=1)
    count = dataset.groupby('Output_label')[data_cols].count().sort_index()
    num_subplots = len(ave)
    fig,ax =  plt.subplots(1,num_subplots, sharey=True)
    fig.set_size_inches(num_subplots*3,3)
    fig.subplots_adjust(top=0.7)
    beat_titles = ['(N), ', '(S), ', '(V), ', '(F), ']
    #beat_titles = ['(N), ', '(S), ', '(V), ', '(F), ', '(Q), ']
    if num_subplots>1:
        for i in range(num_subplots):
            ax[i].plot(np.arange(0,len(data_cols)), ave.iloc[i,:])
            lower_bound = ave.iloc[i,:]-stdev.iloc[i,:]
            upper_bound = ave.iloc[i,:]+stdev.iloc[i,:]
            ax[i].fill_between(np.arange(0,len(data_cols)), lower_bound, upper_bound, facecolor='cornflowerblue', alpha=0.25)
            title = beat_titles[int(ave.index.tolist()[i])] + str(count.values[i,0]) + ', ' +str(round(ave_stdev.values[i],3))
            ax[i].set_title(title)
            x_axis = ax[i].axes.get_xaxis()
            x_axis.set_visible(False)
            if i==0:
                ax[i].set_ylabel('MLII signal (mV)')
                
    else:
        ax.plot(np.arange(0,len(data_cols)), ave.iloc[0,:])
        lower_bound = ave-stdev
        upper_bound = ave+stdev
        ax.fill_between(np.arange(0,len(data_cols)), lower_bound.iloc[0,:], upper_bound.iloc[0,:], facecolor='cornflowerblue', alpha=0.25)
        title = beat_titles[int(ave.index.tolist()[0])] + str(count.values[0,0]) + ', ' +str(round(ave_stdev.values[0],3))
        ax.set_title(title)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        ax.set_ylabel('MLII signal (mV)')
    title = "Average and 1 sigma range per class in the "+dataset_name+'.\nAverage standard deviation of all beats: '\
        +str(round(total_stdev_ave,3))+'.\nClass, count, average sigma for each class'    
    fig.suptitle(title)
    
    
def lda_analysis(dataset, dataset_name):
    X = dataset.drop(columns = ['Patient_number', 'Label', 'Label_class', 'Output_label'], axis=1)
    y = dataset['Output_label']
    target_names = ['N', 'S', 'V', 'F']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit(X_scaled, y).transform(X)
    
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'green']
    
    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.5, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of '+dataset_name+' dataset')
    
def pca_analysis(dataset, dataset_name):
    X = dataset.drop(columns = ['Patient_number', 'Label', 'Label_class', 'Output_label'], axis=1)
    y = dataset['Output_label']
    target_names = ['N', 'S', 'V', 'F']
    pca = PCA(n_components=20)
    X_r = pca.fit(X).transform(X)
    plt.bar(list(range(1,21)), pca.explained_variance_ratio_)
    explained_var = np.sum(pca.explained_variance_ratio_)
    title = "Total explained variance of the 20 PCs: {}".format(round(explained_var,3))
    plt.title(title)
    plt.xlabel('Principal components')
    plt.ylabel('Fraction of explained variance')
    plt.show()
    plt.plot(list(range(1,21)), pca.explained_variance_, ls='--')
    plt.xlabel('Number of principal components')
    plt.ylabel('Eigenvalue')
    plt.title('Scree plot')
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'green']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.5, lw=lw,
                    label=target_name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of '+dataset_name+' dataset')
    plt.tight_layout()
    plt.show()
    loadings = pd.DataFrame(pca.components_.T, columns=list(range(1,21)),index=list(range(0,360)))
    loadings.plot()
    plt.title("Loading scores of the original features")
    plt.xlabel('heartbeat signal points')
    plt.ylabel('Loading scores')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()