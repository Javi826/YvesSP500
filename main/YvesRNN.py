#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

import os
import time
import numpy as np
import pandas as pd
import psutil

import yfinance as yf

from modules.mod_init import *
from paths.paths import file_df_data,folder_csv,path_file_csv
from columns.columns import columns_csv_yahoo,columns_clean_order
from functions.def_functions import set_seeds, class_weight,plots_histograms,plot_loss, plot_accu
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing

from pprint import pprint
from pylab import plt, mpl
from pathlib import Path

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.optimizers.legacy import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_curve, roc_auc_score

start_time = time.time()

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

#LOOPs 1
#------------------------------------------------------------------------------

features =1
lags_val = [50]
f_start_date = '2000-01-01'
f_endin_date = '2018-12-31'
cutoff_dates = ['2017-12-31']

df_results = []

for lags in lags_val:
    for cutoff_date in cutoff_dates:
        print(f"Starts Processing for lags = {lags} and cutoff_date = {cutoff_date}")
        print('\n')
        
        #CALL PREPROCESSING
        #------------------------------------------------------------------------------
        df_preprocessing = mod_preprocessing(df_data_clean,f_start_date,f_endin_date,lags)
        df_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
        df_date_lag_dir = df_preprocessing[df_columns].copy()
               
        #DATA SPLIT
        #------------------------------------------------------------------------------      
        train_data = df_date_lag_dir[df_date_lag_dir['date'] <= cutoff_date]
        tests_data = df_date_lag_dir[df_date_lag_dir['date']  > cutoff_date]
        
        lag_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('lag')]
        
        #X_TRAIN & y_train | NORMALIZATION + RESHAPE
        #------------------------------------------------------------------------------
        X_df_lag_tr = train_data[lag_columns_selected]
        
        scaler_tr = StandardScaler()
        X_df_lag_tr_nr = scaler_tr.fit_transform(X_df_lag_tr)
        X_df_lag_tr_nr = pd.DataFrame(X_df_lag_tr_nr, columns=lag_columns_selected)
               
        X_df_lag_tr_nr_reshaped = X_df_lag_tr_nr.values.reshape(-1, lags, features)
        
        X_train = X_df_lag_tr_nr_reshaped
        y_train = train_data['direction']
        
        #X_TEST & y_tests | NORMALIZATION + RESHAPE
        #------------------------------------------------------------------------------
        X_df_lag_ts = tests_data[lag_columns_selected]
        
        scaler_ts = StandardScaler()
        X_df_lag_ts_ns = scaler_ts.fit_transform(X_df_lag_ts)
        X_df_lag_ts_ns = pd.DataFrame(X_df_lag_ts_ns, columns=lag_columns_selected)
        
        X_df_lag_ts_ns_reshaped = X_df_lag_ts_ns.values.reshape(-1, lags, features)
        
        X_test = X_df_lag_ts_ns_reshaped
        y_test = tests_data['direction']
        
        #LOOPs 2
        #------------------------------------------------------------------------------
        dropout_val = [0.1]
        neurons_val = [30]
        batch_s_val = [16]
        le_rate_val = [0.001]
        optimizers = 'adam'
        #optimizers_to_try = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
        
        for dropout in dropout_val:
            for n_neurons in neurons_val:
                for batch_s in batch_s_val:
                    for le_rate in le_rate_val:
                        print(f"Training model starts for Dropout = {dropout}, Neurons = {n_neurons}, Batch Size = {batch_s}, Learning Rate = {le_rate}, Optimizer = {optimizers}")
        
                        set_seeds()
                        model = Sequential()
                        model.add(SimpleRNN(n_neurons, input_shape=(lags, features), return_sequences=True))
                        model.add(Dropout(dropout))
                        model.add(SimpleRNN(n_neurons))
                        model.add(Dense(1, activation='sigmoid'))
        
                        optimizer = Adam(learning_rate=le_rate)
                        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                        
                        #path_destiny = Path.cwd()
                        #file_name = 'best_model.h5'
                        #path_file_best = path_destiny / file_name
                        
                        #checkpoint = ModelCheckpoint(path_file_best, 
                        #                             monitor='val_accuracy', 
                        #                             save_best_only=True, 
                        #                             mode='max', 
                        #                             verbose=1)
                        
                        history = model.fit(X_train, y_train, 
                                            epochs=250, 
                                            verbose=0,
                                            validation_data=(X_test, y_test),
                                            batch_size=batch_s)
        
                        # y_pred
                        y_pred = model.predict(X_test, batch_size=None)
                        y_pred_binary = (y_pred > 0.5).astype(int)
                        
                        train_loss = history.history['loss'][-1]
                        train_accu = history.history['accuracy'][-1]
                        
                        # Métricas de validación
                        validation_loss = history.history['val_loss'][-1]
                        validation_accu = history.history['val_accuracy'][-1]
        
                        accuracy  = accuracy_score(y_test, y_pred_binary)
                        precision = precision_score(y_test, y_pred_binary)
                        recall    = recall_score(y_test, y_pred_binary)
                        f1        = f1_score(y_test, y_pred_binary)
                        auc_roc   = roc_auc_score(y_test, y_pred)
        
                        df_results.append({ 'Lags         ': lags,
                                            'Cutoff Date  ': cutoff_date,
                                            'Dropout      ': dropout,
                                            'Neurons      ': n_neurons,
                                            'Batch Size   ': batch_s,
                                            'Learning Rate': le_rate,
                                            'Optimizer    ': optimizers,
                                            'Train Loss   ': train_loss,
                                            'Val Loss     ': validation_loss,
                                            'Train Accu   ': train_accu,
                                            'Val Accu     ': validation_accu,
                                            'Accuracy     ': accuracy,
                                            'Precision    ': precision,
                                            'Recall       ': recall,
                                            'F1-Score     ': f1,
                                            'AUC-ROC      ': auc_roc})
                        print(f"Training model ending for Dropout = {dropout}, Neurons = {n_neurons}, Batch Size = {batch_s}, Learning Rate = {le_rate}, Optimizer = {optimizers}")
                        print('\n')
        
                        
        #plot_loss(history)
        plot_accu(history)
        
        print(f"Ending Processing for lags = {lags} and cutoff_date = {cutoff_date}")
        print('\n')
        
df_results_all = pd.DataFrame(df_results)

# Guarda el DataFrame en un archivo Excel
df_results_all.to_excel('df_results_all.xlsx', index=False)
print("All results saved in: 'df_results_all.xlsx'")

elapsed_time = time.time() - start_time
hours, minutes = divmod(elapsed_time, 3600)
minutes = minutes / 60  

os.system("afplay /System/Library/Sounds/Ping.aiff")
print(f"Total time taken for the process: {int(hours)} hours, {int(minutes)} minutes")
