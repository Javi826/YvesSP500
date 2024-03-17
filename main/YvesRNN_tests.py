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


#TEST DATA SELECTION ON PREPROCESSING file
#------------------------------------------------------------------------------
lags = 5
features =1
start_tests = '2000-01-01'
endin_tests = '2019-12-31'

file_suffix = f"_{str(lags).zfill(2)}_{start_tests}_{endin_tests}.xlsx"
path_preprocessing = os.path.join(path_base, folder_preprocessing, f"df_preprocessing{file_suffix}")
df_preprocessing = pd.read_excel(path_preprocessing, header=0, skiprows=0)

df_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
df_date_lag_dir = df_preprocessing[df_columns].copy()

#TESTS DATA SPLIT
#------------------------------------------------------------------------------    
start_cutoff_tests  = '2018-01-01'
endin_cutoff_tests  = '2018-12-31'

tests_data = df_date_lag_dir[(df_date_lag_dir['date'] >= start_cutoff_tests) & (df_date_lag_dir['date'] <= endin_cutoff_tests)]
lag_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('lag')]

#X_TEST & y_test | NORMALIZATION + RESHAPE
#------------------------------------------------------------------------------
X_df_lag_ts = tests_data[lag_columns_selected]

scaler_ts = StandardScaler()
X_df_lag_ts_nr = scaler_ts.fit_transform(X_df_lag_ts)
X_df_lag_ts_nr = pd.DataFrame(X_df_lag_ts_nr, columns=lag_columns_selected)

X_df_lag_ts_nr_reshaped = X_df_lag_ts_nr.values.reshape(-1, lags, features)

X_tests = X_df_lag_ts_nr_reshaped
y_tests = tests_data['direction']

#LOAD BEST MODEL
#------------------------------------------------------------------------------
dropout   = 0.1
n_neurons = 30
batch_s   = 16
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


