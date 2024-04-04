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

from paths.paths import path_preprocessing,path_base,folder_preprocessing,results_path, path_base, folder_tests_results,tf_serving_path,folder_tf_serving
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_curve, roc_auc_score
from keras.models import load_model
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from modules.mod_pipeline import mod_pipeline


#TEST DATA SELECTION ON PREPROCESSING file
#------------------------------------------------------------------------------
lags = 20
n_features =1
prepro_start_date = '2000-01-01'
prepro_endin_date = '2019-12-31'

file_suffix = f"_{str(lags).zfill(2)}_{prepro_start_date}_{prepro_endin_date}.xlsx"
path_preprocessing = os.path.join(path_base, folder_preprocessing, f"df_preprocessing{file_suffix}")
df_preprocessing = pd.read_excel(path_preprocessing, header=0, skiprows=0)
#print(df_preprocessing)

df_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
df_date_lag_dir = df_preprocessing[df_columns].copy()

#LOAD BEST MODEL + SAVE MODEL FOR TF SERVING
#------------------------------------------------------------------------------
dropout   = 0.1
n_neurons = 30
batch_s   = 32
le_rate   = 0.001
endin_data_valid  = '2018-12-31'

file_model_name = f'model_lags_{str(lags).zfill(2)}_date_{endin_data_valid}_dropout_{dropout}_neurons_{n_neurons}_batch_{batch_s}_lr_{le_rate}.keras'
best_model_path = os.path.join(results_path, file_model_name)
best_model      = load_model(best_model_path)

tf.saved_model.save(best_model, tf_serving_path)

#TESTS DATA SPLIT
#------------------------------------------------------------------------------    
initn_data_tests  = ['2010-01-01']
endin_data_tests  = '2018-12-31'

#X_TEST & y_test | NORMALIZATION + RESHAPE
#------------------------------------------------------------------------------
X_tests = mod_pipeline(df_preprocessing, initn_data_tests, endin_data_tests,lags, n_features, 'X_tests')
y_tests = mod_pipeline(df_preprocessing, initn_data_tests, endin_data_tests,lags, n_features, 'y_tests')

#MODEL PREDICTIONS
#------------------------------------------------------------------------------
y_pred = best_model.predict(X_tests)
y_pred_bin = (y_pred > 0.5).astype(int)

#MODEL PREDICTIONS METRICS
#------------------------------------------------------------------------------

accuracy  = accuracy_score(y_tests, y_pred_bin)
f1        = f1_score(y_tests, y_pred_bin)
recall    = recall_score(y_tests, y_pred_bin)
precision = precision_score(y_tests, y_pred_bin)

print("Tests_Accuracy :", accuracy)
print("Tests_F1 Score :", f1)
print("Tests_Recall   :", recall)
print("Tests_Precision:", precision)

tests_results = {
    'initn_data_tests': [initn_data_tests],
    'endin_data_tests': [endin_data_tests],
    'dropout         ': [dropout],
    'n_neurons       ': [n_neurons],
    'batch_s         ': [batch_s],
    'le_rate         ': [le_rate],
    'Tests_Accuracy  ': [accuracy],
    'Tests_F1_Score  ': [f1],
    'Tests_Recall    ': [recall],
    'Tests_Precision ': [precision]
}

# Convierte el diccionario en un DataFrame
df_tests_results = pd.DataFrame(tests_results)

excel_file_path = os.path.join(path_base, folder_tests_results, f"df_tests_results_all.xlsx")
df_tests_results.to_excel(excel_file_path, index=False)
print("All Training results saved in: 'tra_val_results/df_tests_results_all.xlsx'")





