#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 01:12:23 2024
@author: javi
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from paths.paths import path_preprocessing,path_base,folder_preprocessing,results_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from modules.mod_pipeline import mod_pipeline


#TEST DATA SELECTION ON PREPROCESSING file
#------------------------------------------------------------------------------
lags = 5
n_features =1
start_tests = '2000-01-01'
endin_tests = '2019-12-31'

file_suffix = f"_{str(lags).zfill(2)}_{start_tests}_{endin_tests}.xlsx"
path_preprocessing = os.path.join(path_base, folder_preprocessing, f"df_preprocessing{file_suffix}")
df_preprocessing = pd.read_excel(path_preprocessing, header=0, skiprows=0)

df_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
df_date_lag_dir = df_preprocessing[df_columns].copy()

#TESTS DATA SPLIT
#------------------------------------------------------------------------------    
initn_data_tests  = ['2018-01-01']
endin_data_tests  = '2019-12-31'

#X_TEST & y_test | NORMALIZATION + RESHAPE
#------------------------------------------------------------------------------
X_tests = mod_pipeline(df_preprocessing, initn_data_tests, endin_data_tests,lags, n_features, 'X_tests')
y_tests = mod_pipeline(df_preprocessing, initn_data_tests, endin_data_tests,lags, n_features, 'y_tests')


#LOAD BEST MODEL
#------------------------------------------------------------------------------
dropout   = 0.1
n_neurons = 30
batch_s   = 32
le_rate   = 0.001
cutoff_train  = '2017-01-01'

file_model_name = f'model_lags_{str(lags).zfill(2)}_date_{cutoff_train}_dropout_{dropout}_neurons_{n_neurons}_batch_{batch_s}_lr_{le_rate}.h5'
best_model_path = os.path.join(results_path, file_model_name)
best_model = load_model(best_model_path)

#MODEL PREDICTIONS
#------------------------------------------------------------------------------
y_pred = best_model.predict(X_tests)
y_pred_bin = (y_pred > 0.5).astype(int)

#MODEL PREDICTIONS METRICS
#------------------------------------------------------------------------------
accuracy    = accuracy_score(y_tests, y_pred_bin)
conf_matrix = confusion_matrix(y_tests, y_pred_bin)

# Imprimir resultados
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)


