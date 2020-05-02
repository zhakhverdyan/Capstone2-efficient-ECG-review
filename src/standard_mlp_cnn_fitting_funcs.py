#!/anaconda3/envs mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:02:22 2020

@author: zhannahakhverdyan

The functions in this script produce convolutional are fully connected networks of desired
architecture, train and tune the network with train and validation data respectively and
produce a confusion matrix for the test data. Use this script in a loop to systematically 
test network parameters.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
 
def process_data(train, val, test, model_type, with_standard=False):
    """Take the train, val and test datasets as pandas dataframe 
    and process for MLP or CNN fitting. If with_standard = True, use standardization"""
    x_train = train.drop(columns=['Patient_number', 'Label', 'Label_class', 'Output_label']).values
    val.loc[val['Label_class']=='N', 'Weight'] = 1
    val.loc[val['Label_class']=='S', 'Weight'] = 70
    val.loc[val['Label_class']=='V', 'Weight'] = 14
    val.loc[val['Label_class']=='F', 'Weight'] = 493
    x_val = val.drop(columns=['Patient_number', 'Label', 'Label_class', 'Output_label', 'Weight']).values
    x_test = test.drop(columns=['Patient_number', 'Label', 'Label_class', 'Output_label']).values
    val_sample_weight = val.Weight.values
    if model_type=='cnn':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_train = to_categorical(train.Output_label)
    y_val = to_categorical(val.Output_label)
    y_test = to_categorical(test.Output_label)
    if with_standard==True:
        scaler = StandardScaler(with_mean=False,with_std=False)
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    label = train.Output_label.values
    class_weights = class_weight.compute_class_weight('balanced', np.unique(label), label)
    class_weights = class_weights/min(class_weights)
    class_weights = [round(class_weight) for class_weight in class_weights]
    print("Class weights", list(zip(class_weights, np.unique(label))))
    return x_train, x_val, x_test, y_train, y_val, y_test, class_weights, val_sample_weight

def build_model_mlp(l, n, d, x_train, model_prefix):
    """l+1=number of layers, n=number of nodes, d=dropout rate"""
    # build the model
    model = Sequential()
    model.add(Dense(n, activation = 'relu', input_dim = x_train.shape[1]))
    model.add(Dropout(rate = d))
    for i in range(l):
        model.add(Dense(n, activation = 'relu'))
        model.add(Dropout(rate = d))
    model.add(Dense(4, activation = 'softmax')) # eliminated Q label
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('../../models/{}best_model.h5'.format(model_prefix), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #compile the model, use categorical cross entropy and Adam optimizer
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = [sClassRecall, f1_metric, 'categorical_accuracy'])
    return model, es, mc, f1_metric

def sClassRecall(y_true, y_pred):
    true_positive_s = K.sum(K.round(K.clip(y_true[:,1] * y_pred[:,1], 0, 1)))
    possible_positive_s = K.sum(K.round(K.clip(y_true[:,1], 0, 1)))
    recall_s = true_positive_s / (possible_positive_s + K.epsilon())
    return recall_s

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:,1:] * y_pred[:,1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:,1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:,1:], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def avePrecision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:,1:] * y_pred[:,1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:,1:], 0, 1)))
    ave_prec = (true_positives/predicted_positives)
    return ave_prec

def build_model_cnn(l,n,d, k, x_train, model_prefix):
    """Build a convolution nn with l+1 layers, n nodes, max norm coefficient 4
    and d droput rate"""
    model = Sequential()
    model.add(Dropout(rate = d, input_shape = (x_train.shape[1], 1)))
    model.add(Conv1D(filters=n, kernel_size=(k), strides=1, kernel_constraint=max_norm(4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    for i in range(l):
        model.add(Conv1D(filters=n, kernel_size=(k), strides=1, kernel_constraint=max_norm(4), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(rate = d))
        model.add(MaxPooling1D(pool_size=k))
    model.add(Flatten())
    model.add(Dense(n, kernel_constraint=max_norm(4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(4, activation = 'softmax')) # eliminated Q label
    es = EarlyStopping(monitor='val_avePrecision', mode='max', verbose=1, patience=10)
    mc = ModelCheckpoint('../../models/{}best_model.h5'.format(model_prefix), monitor='val_avePrecision', mode='max', verbose=1, save_best_only=True)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',\
                 metrics = [sClassRecall, f1_metric, 'categorical_accuracy'])
    # to penalize the model based on the prediction probability as apposed to absolute predictions
    # uncomment the model.compile statement below and xomment out the model.compile statement
    # above
    #model.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=losses_utils.ReductionV2.SUM),\
    #              optimizer = 'adam', metrics = [avePrecision, 'categorical_accuracy'])
    return model, es, mc

def fit_model(model, x_train, y_train, x_val, y_val, class_weights, es, mc, val_sample_weight):
    history = model.fit(x_train, y_train, validation_data = [x_val, y_val, val_sample_weight], shuffle=True, batch_size=32, epochs = 100, \
                    verbose=1, class_weight=class_weights, callbacks=[es, mc])
    return history

def plot_confusion_matrix(model, x_test, y_test, model_prefix, normalize=True):
    """
    This function plots the confusion matrix. Default normalization is true.
    To get raw values set `normalize=False`.
    """
    # compute probabilities for each class
    y_pred = model.predict_proba(x_test, batch_size=32, verbose=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    np.set_printoptions(precision=2)
    
    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"+model_prefix
    else:
        title = "Confusion matrix, without normalization"+model_prefix

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    #classes=['N', 'S', 'V', 'F', 'Q']
    classes=['N', 'S', 'V', 'F']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #return cm

def full_process(train_df, val_df, test_df, l, n, d, k, model_type, with_standard=False):
    model_prefix = '%d_%d_%.2f_%d_' % (l, n, d, k)
    x_train, x_val, x_test, y_train, y_val, y_test, class_weights, val_sample_weight = \
        process_data(train_df, val_df, test_df, model_type, with_standard=False)
    if model_type=='cnn':
        model, es, mc = build_model_cnn(l,n,d, k, x_train, model_prefix)
    elif model_type=='mlp':
        model, es, mc, f1_metric = build_model_mlp(l,n,d, x_train, model_prefix)
    # plot training history
    history = fit_model(model, x_train, y_train, x_val, y_val, class_weights, es, mc, val_sample_weight)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    title = model_prefix+'history'
    plt.title(title)
    plt.legend()
    plt.show()
    saved_model = load_model('../../models/{}best_model.h5'.format(model_prefix), custom_objects={'avePrecision': avePrecision})
    train_loss, train_ave_prec, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_ave_prec, val_acc = saved_model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_ave_prec, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)
    print('Train loss: %.3f, Val loss: %.3f, Test loss: %.3f' % (train_loss, val_loss, test_loss))
    print('Train ave prec: %.3f, Val ave prec: %.3f, Test ave prec: %.3f' % (train_ave_prec, val_ave_prec, test_ave_prec))
    print('Train accuracy: %.3f, Val accuracy: %.3f, Test accuracy: %.3f' % (train_acc, val_acc, test_acc))
    # compute and plot the confusion matrix
    plot_confusion_matrix(saved_model, x_test, y_test, model_prefix, normalize=True)