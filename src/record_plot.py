#!/anaconda3/envs mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:31:17 2020

@author: zhannahakhverdyan

The functions in this script are used to produce the plotly graphs for result visualization.
"""

import wfdb
import numpy as np
import plotly.graph_objects as go


path = '../../mit-bih-arrhythmia-database-1.0.0/'
def pull_signal_4abs_plotting(record):
    """"Pull the signals, beat annotations and coordinates from waveform database"""
    signals, fields = wfdb.rdsamp(path+record, channels=[0])
    annotation = wfdb.rdann(path+record, 'atr', return_label_elements=['symbol'],\
                        summarize_labels=True)
    annot_symbols = annotation.symbol
    annot_x_coord = annotation.sample.tolist()
    annot_x_coord = [x/360 for x in annot_x_coord] # translates coordinate unit to seconds
    return signals, annot_symbols, annot_x_coord

def labelTranslation(annot_symbols):
    """Encole beat labels into 4 major classes"""
    annot_symbols_relabeled = []
    for label in annot_symbols:
        if label in ['N', 'L', 'R', 'e', 'j']:
            annot_symbols_relabeled.append('N')
        elif label in ['A', 'a', 'J', 'S']:
            annot_symbols_relabeled.append('S')
        elif label in ['V', 'E']:
            annot_symbols_relabeled.append('V')
        elif label=='F':
            annot_symbols_relabeled.append('F')
        else:
            annot_symbols_relabeled.append('')
    return annot_symbols_relabeled
        
def plotRecord(annot_symbols_relabeled, annot_x_coord, test_df_pred, record, signals):
    """Plot a plotly graph with the raw trace and abnormal beat predicitons with corresponding probabilities"""
    gt = [(label, coord) for (label, coord) in zip(annot_symbols_relabeled, annot_x_coord) \
                           if label in ['N','S', 'V','F']]
    
    pred = test_df_pred[(test_df_pred['Patient_number']==int(record))]['Label_pred'].values
        
    prob = test_df_pred[(test_df_pred['Patient_number']==int(record))]['Probability'].values
    gt_coord_pred_prob = [(label[0], label[1], label_pred, prob) for (label, label_pred, prob) in \
                       zip(gt, pred, prob) if label[0] in ['S', 'V', 'F'] or label_pred  in ['S', 'V', 'F']]
    
    y=np.squeeze(signals)
    y=y[::6] # downsample y 6-fold, comment out this line to plot full data
    x = list(range(len(y)))  
    x = [coord/60 for coord in x] # use  x = [val/360 for val in x] for full dataset
    
    predicted_sum = test_df_pred[test_df_pred['Patient_number']==int(record)]['Label_pred'].value_counts()
    total_count = predicted_sum.sum()
    fig = go.Figure(data=go.Scatter(x=x, y=y, name="Raw ECG signal ({} total beats)".format(total_count),\
                                    marker=dict(color='grey')))
    
    fig.update_layout(title_text="{} record signal with labeled abnormal beats".format(record),\
                      title_font_size=24, xaxis_title="Time (sec)",\
                      yaxis_title="ECG signal (mV)", xaxis_showgrid=True, xaxis_gridcolor='rgba(31, 119, 180)',\
                      yaxis_showgrid=True, yaxis_gridcolor='rgba(31, 119, 180)')

    def annotSelector(label_class, label_list, p):
        pred_coord_l = [record[1] for record in label_list if record[2]==label_class and record[3]<p]
        pred_coord_h = [record[1] for record in label_list if record[2]==label_class and record[3]>=p]
        return pred_coord_l, pred_coord_h
    
    pred_s_l, pred_s_h = annotSelector('S', gt_coord_pred_prob, 0.95)
    pred_v_l, pred_v_h = annotSelector('V', gt_coord_pred_prob, 0.95)
    pred_f_l, pred_f_h = annotSelector('F', gt_coord_pred_prob, 0.95)
    pred_n_l, pred_n_h = annotSelector('N', gt_coord_pred_prob, 0.6)
    
    gt_label = [record[0] for record in gt_coord_pred_prob]
    gt_coord = [record[1] for record in gt_coord_pred_prob]
    
    
    def addTrace(label_coord, color, height, label_class, group, visibility):
        fig.add_trace(go.Scatter(
        x=label_coord,
        y=[height]*len(label_coord),
        visible=visibility,
        mode="markers+text",
        legendgroup="group"+group,  # this is for high or low confidence
        name="{} prediction (count: {}, confidence: {})".format(label_class, len(label_coord), group),
        text=[label_class]*len(label_coord),
        textposition="bottom center",
        marker=dict(color=color)))

    if 'S' in predicted_sum: 
        addTrace(pred_s_h, 'red', 3, 'S', 'high', True)
        addTrace(pred_s_l, 'red', 3, 'S', 'low', 'legendonly')
    if 'V' in predicted_sum:   
        addTrace(pred_v_h, 'blue', 3, 'V', 'high', True)
        addTrace(pred_v_l, 'blue', 3, 'V', 'low', 'legendonly')
    if 'F' in predicted_sum:  
        addTrace(pred_f_h, 'purple', 3, 'F', 'high', True)
        addTrace(pred_f_l, 'purple', 3, 'F', 'low', 'legendonly')
    addTrace(pred_n_l, 'orange', 3, 'N', 'suspected FN', 'legendonly')
    
    fig.add_trace(go.Scatter(
        x=gt_coord,
        y=[-3]*len(gt_coord),
        visible='legendonly',
        mode="text",
        name="True labels",
        text=gt_label,
        textposition="bottom center",
    ))

    return fig # returns a figure object to return a figure use fig.show()
  

def recordPlot(record, test_df_pred):
    signals, annot_symbols, annot_x_coord = pull_signal_4abs_plotting(record)
    annot_symbols_relabeled = labelTranslation(annot_symbols)
    fig = plotRecord(annot_symbols_relabeled, annot_x_coord, test_df_pred, record, signals)
    return fig
