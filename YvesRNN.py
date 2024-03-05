#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf

from modules.mod_init import *
from paths.paths import file_df_data,folder_csv,path_file_csv
from columns.columns import columns_csv_yahoo,columns_clean_order
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from functions.def_functions import class_weight,create_deep_rnn_model,set_seeds

from pprint import pprint
from pylab import plt, mpl

from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense,Dropout
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_curve, roc_auc_score

lags=5
features=1
# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
symbol = "^GSPC"
start_date = "1980-01-01"
endin_date = "2023-12-31"
sp500_data = yf.download(symbol, start=start_date, end=endin_date)
sp500_data.to_csv(path_file_csv)
df_data = pd.read_csv(path_file_csv, header=None, skiprows=1, names=columns_csv_yahoo)

#CALL module Datacleaning
#------------------------------------------------------------------------------

df_data_clean = mod_dtset_clean(df_data,start_date,endin_date)
#print('df_data_clean:')
#print(df_data_clean)

#CALL PREPROCESSING
#------------------------------------------------------------------------------

filter_start_date = '2000-01-01'
filter_endin_date = '2018-12-31'
df_preprocessing = mod_preprocessing(df_data_clean,filter_start_date,filter_endin_date)
#print('df_preprocessing:')
#print(df_preprocessing)

lag_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
df_lag_dir = df_preprocessing[lag_columns].copy()
#print('df_lag_dir:')
#print(df_lag_dir)


#SPLIT
split = int(len(df_lag_dir) * 0.8)
lag_columns_selected = [col for col in df_lag_dir.columns if col.startswith('lag')]

# X_TRAIN y_train
X_df_lag_tr = df_lag_dir[lag_columns_selected].iloc[:split].copy()

#print('X_df_lag_tr:')
#print(X_df_lag_tr)

mu_tr, std_tr = X_df_lag_tr.mean(), X_df_lag_tr.std()

X_df_lag_tr_nr = (X_df_lag_tr - mu_tr) / std_tr
#print('X_df_lag_tr_nr:')
#print(X_df_lag_tr_nr)

X_df_lag_tr_nr_reshaped = X_df_lag_tr_nr.values.reshape(-1, 5, 1)


y_train = df_lag_dir['direction'].iloc[:split].copy()
#print('y_train:')
#print(y_train)

#X_TEST y_test

X_df_lag_ts = df_lag_dir[lag_columns_selected].iloc[split:].copy()

mu_ts, std_ts = X_df_lag_ts.mean(), X_df_lag_ts.std()

X_df_lag_ts_ns = (X_df_lag_ts - mu_ts) / std_ts

y_test = df_lag_dir['direction'].iloc[split:].copy()
#print('y_test:')
#print(y_test)

lags=5
features=1
set_seeds()

dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#df_results = pd.DataFrame(columns=['Dropout', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
df_results = []

for dropout_rate in dropout_values:
    print(f"Training model starts for Dropout = {dropout_rate}")

    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(lags, features), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_df_lag_tr_nr_reshaped, y_train, epochs=50, verbose=0,
              validation_data=(X_df_lag_ts_ns.values.reshape(-1, 5, 1), y_test))

    #y_pred
    y_pred = model.predict(X_df_lag_ts_ns.values.reshape(-1, 5, 1), batch_size=None)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    df_results.append({'Dropout': dropout_rate,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1,
                            'AUC-ROC': auc_roc})

    print("Training model ending for Dropout =", dropout_rate)
    print("-" * 54)  # Línea divisoria para mejorar la legibilidad en la salida


df_results = pd.DataFrame(df_results)

# Guardar resultados en un archivo Excel
df_results.to_excel('metrics_results.xlsx', index=False)
print("Results saved in: 'metrics_results.xlsx'")
