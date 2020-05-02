#!/anaconda3/envs mit-bih-arrhythmia-database-1.0.0
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:45:42 2020

@author: zhannahakhverdyan

Run tests on feature extraction functions
"""
import numpy as np
import pandas as pd

import feature_extraction_funcs as fef

def test_dif_next_point():
    x = np.array([0, 1, 2])
    x_dif = fef.dif_next_point(x)
    assert np.allclose(x_dif, np.array([1, 1]))
    
def test_group_labels():
    df = pd.DataFrame({'column1':[0,0,0], 'Label':['L', 'E', 'f']})
    df_processed = fef.group_labels(df)
    assert df_processed.Label_class.tolist() == ['N', 'V', 'Q'], "Grouping not successful"

def test_code_labels():
    df = pd.DataFrame({'column1':[0,0,0], 'Label_class':['N', 'V', 'F']})
    df_processed = fef.code_labels(df)
    assert df_processed.Output_label.tolist() == [0, 2, 3], "Label encoding not sucessful"           

def test_extract_labeled_heartbeat():
    file_name = '100'
    signal = np.arange(0,21,1)
    samp_index = [2, 19]
    annot_symbol = ['N', 'F']
    window = 0.1
    df = fef.extract_labeled_heartbeat(file_name, signal, samp_index, annot_symbol, window)
    assert df.values.shape == (2, 38), "wrong heartbeat extraction"
    assert df.Patient_number.unique() == 100, "wrong patient number assignment"
    assert df.Label.tolist() == ['N', 'F'], "wrong labeling"