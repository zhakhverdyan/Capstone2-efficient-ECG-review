#!/anaconda3/envs mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:51:34 2020

@author: zhannahakhverdyan

The functions in this script are used to evaluate the model performance wither on one record
or the whole dataset. 
"""
import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical
#import pandas as pd
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.constraints import max_norm

def evaluate1patient(df, patient):
    """This function is used for N-1 cross-validation. While looping through patient numbers
    provide the whole dataset (train+val+test) and the patient number. The dataset is split 
    into test (patient record data) and train (everything but the patient record data). After
    training and evaluation the history, evaluation array and confusion matrix are returned.
    Note, the evaluation array and confusion matrix must be collected into a list for oveeral
    summary."""
    test_1pat = df[df['Patient_number']==patient]
    x_test, y_test, target_unique = prepTestSet(test_1pat)
    remaining_pateients = df[df['Patient_number']!=patient]
    x_train, y_train = prepTrain(remaining_pateients)
    label = remaining_pateients.Output_label.values
    class_weights = class_weight.compute_class_weight('balanced', np.unique(label), label)
    class_weights = class_weights/min(class_weights)
    class_weights = [round(class_weight) for class_weight in class_weights]
    model = build_model_cnn(x_train, n=32, d=0.25, k=5)
    history=fitModel(model, x_train, y_train, class_weights)
    return_array, cm = modelEvaluator(x_test, y_test, model, target_unique)
    return history, return_array, cm

def prepTestSet(test_1pat):
    x_test = test_1pat.drop(columns=['Unnamed: 0', 'Patient_number', 'Label', 'Label_class', 'Output_label']).values
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    target_unique = test_1pat.Output_label.unique()
    target_unique.sort(axis=0)
    target_unique = target_unique.astype('int')
    # always return 4 target column irrespecitive of number of labels in the sample
    y_test = np.zeros((len(test_1pat),4))
    for target in target_unique:
        test_1pat.loc[test_1pat['Output_label'] == target, 'one_hot_label'] = 1
        test_1pat.loc[test_1pat['Output_label'] != target, 'one_hot_label'] = 0
        target_col = test_1pat.one_hot_label.values
        y_test[:,int(target)] = target_col
    test_1pat.drop('one_hot_label', axis=1, inplace=True)
    return x_test, y_test, target_unique

def prepTrain(remaining_pateients):
    x_train = remaining_pateients.drop(columns=['Unnamed: 0', 'Patient_number', 'Label', 'Label_class', 'Output_label']).values
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = to_categorical(remaining_pateients.Output_label)
    return x_train, y_train

def build_model_cnn(x_train, n=32, d=0.25, k=5):
    """Build a convolution nn with l+1 layers, n nodes, max norm coefficient 4
    and d droput rate"""
    model = Sequential()
    model.add(Dropout(rate = d, input_shape = (x_train.shape[1], 1)))
    model.add(Conv1D(filters=n, kernel_size=(5), strides=1, activation = 'relu', kernel_constraint=max_norm(4)))
    model.add(Dropout(rate = d))
    model.add(MaxPooling1D(pool_size=k))
    model.add(Conv1D(filters=n, kernel_size=(5), strides=1, activation = 'relu', kernel_constraint=max_norm(4)))
    model.add(Dropout(rate = d))
    model.add(MaxPooling1D(pool_size=k))
    model.add(Flatten())
    model.add(Dense(n, activation ='relu', kernel_constraint=max_norm(4)))
    model.add(Dense(4, activation = 'softmax')) # eliminated Q label
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
    return model  

def fitModel(model, x_train, y_train, class_weights):
    history = model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs = 10, \
                    verbose=1, class_weight=class_weights)
    return history


def modelEvaluator(x_test, y_test, model, target_unique):
"""Evaluates model performance for each beat class individually and for overall label class
differentiation as defined by AAMI
Input: test dataset, ground truth (one hot encoded), the model to be evaluated and list of unique
labels present in the dataset.
All function inputs except from the model are prepared with the prepTestSet() function.
Output: array containing individual metrics and the confusion matrix."""
    epsilon =  np.finfo(float).eps
    # predict y_pred
    y_pred = model.predict_proba(x_test, batch_size=32, verbose=1)
    # calculate the confusion matrix
    cm = tf.math.confusion_matrix(tf.math.argmax(y_test, 1), tf.math.argmax(y_pred, 1), num_classes=4)
    cm = cm.numpy()
    # actual normal
    Nn = cm[0,0]
    Ns = cm[0,1]
    Nv = cm[0,2]
    Nf = cm[0,3]
    # actual Supraventricular
    Sn = cm[1,0]
    Ss = cm[1,1]
    Sv = cm[1,2]
    Sf = cm[1,3]
    # actual Ventricular
    Vn = cm[2,0]
    Vs = cm[2,1]
    Vv = cm[2,2]
    Vf = cm[2,3]
    # actual Fusion
    Fn = cm[3,0]
    Fs = cm[3,1]
    Fv = cm[3,2]
    Ff = cm[3,3]
    
    sum_all = np.sum(cm)
    # metrics for distinguishing ventricular ectopic beats
    TNv = Nn + Ns + Nf + Sn + Ss + Sf + Fn + Fs + Ff # (True negative V)
    FPv = Nv + Sv #(False positive V)
    fprV = FPv / (TNv + FPv + epsilon) #(false positive rate V)
    if 2 in target_unique:
        FNv = Vn + Vs +Vf # (False negative V)
        TPv = Vv #(True positive V)
        SeV = TPv / (TPv + FNv + epsilon) #(sensitivity or recall V)
        PrV = TPv / (TPv + FPv + epsilon) #(positive predictivity or precision V)
        AccV = (TPv + TNv) / (TPv + TNv + FPv + FNv + epsilon) #(Accuracy V)
    else:
        SeV = np.nan
        PrV = np.nan
        AccV = np.nan
    
    
    # metrics for distinguishing supraventricular ectopic beats
    TNs = Nn + Nv + Nf + Vn + Vv + Vf + Fn + Fv + Ff #(True negative S)
    FPs= Ns + Vs + Fs #(False positive S)
    fprS = FPs / (TNs + FPs + epsilon) #(false positive rate S)
    if 1 in target_unique:
        FNs = Sn + Sv +Sf #(False negative S)
        TPs = Ss #(True positive S)
        SeS = TPs / (TPs + FNs + epsilon) #( sensitivity or recall S)
        PrS = TPs / (TPs + FPs + epsilon) #(positive predictivity or precision S)
        AccS = (TPs + TNs) / (TPs + TNs + FPs + FNs + epsilon) #(Accuracy S)
    else:
        SeS = np.nan
        PrS = np.nan
        AccS = np.nan
    
    # metrics for distinguishing all beats
    TN = Nn #(True negative)
    TPf = Ff
    FPN = Ns + Nv + Nf # false positive N class
    Sp = TN/(TN + FPN + epsilon)
    sum_f = Fn + Fs + Fv + Ff
    if 3 in target_unique:
        SeF = TPf/(sum_f + epsilon)
    else:
        SeF = np.nan
    Acc = (Nn + Ss + Vv + Ff)/sum_all
    
    return_array = [round(SeV, 3), round(PrV, 3), round(fprV, 3), round(AccV, 3), round(SeS, 3), round(PrS, 3),\
                    round(fprS, 3), round(AccS, 3), round(SeF, 3), round(Sp, 3), round(Acc, 3)]
    return return_array, cm
