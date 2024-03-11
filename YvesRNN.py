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
from functions.def_functions import class_weight,create_deep_rnn_model,set_seeds
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing

from pprint import pprint
from pylab import plt, mpl

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.optimizers.legacy import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_curve, roc_auc_score


os.system("afplay /System/Library/Sounds/Ping.aiff")

lags=5
features=1
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

#CALL PREPROCESSING
#------------------------------------------------------------------------------
filter_start_date = '2000-01-01'
filter_endin_date = '2018-12-31'
df_preprocessing = mod_preprocessing(df_data_clean,filter_start_date,filter_endin_date,lags)

print(df_preprocessing)


lag_columns =['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
df_lag_dir = df_preprocessing[lag_columns].copy()

#DATA SPLIT
#------------------------------------------------------------------------------
split = int(len(df_lag_dir) * 0.8)
lag_columns_selected = [col for col in df_lag_dir.columns if col.startswith('lag')]

# X_TRAIN y_train
#------------------------------------------------------------------------------
X_df_lag_tr = df_lag_dir[lag_columns_selected].iloc[:split].copy()

mu_tr, std_tr = X_df_lag_tr.mean(), X_df_lag_tr.std()
X_df_lag_tr_nr = (X_df_lag_tr - mu_tr) / std_tr

X_df_lag_tr_nr_reshaped = X_df_lag_tr_nr.values.reshape(-1, lags, 1)

X_train = X_df_lag_tr_nr_reshaped

y_train = df_lag_dir['direction'].iloc[:split].copy()

#X_TEST y_test
#------------------------------------------------------------------------------
X_df_lag_ts = df_lag_dir[lag_columns_selected].iloc[split:].copy()

mu_ts, std_ts = X_df_lag_ts.mean(), X_df_lag_ts.std()
X_df_lag_ts_ns = (X_df_lag_ts - mu_ts) / std_ts

X_df_lag_ts_ns_reshaped = X_df_lag_ts_ns.values.reshape(-1, lags, 1)

X_test = X_df_lag_ts_ns_reshaped

y_test = df_lag_dir['direction'].iloc[split:].copy()

#LOOPs
#------------------------------------------------------------------------------
#dropout_values = [0.1, 0.2, 0.3, 0.4]
#dropout_values = [0.5, 0.6, 0.7]
#dropout_values = [0.8]
#dropout_values = [0.9]

#neurons_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#batch_sizes    = [8, 16, 32, 64, 128]
#learning_rates= [0.1, 0.01, 0.001, 0.0001]
#optimizers_to_try = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]

dropout_values = [0.1]
neurons_values = [10]
batch_sizes = [32]
learning_rates = [0.001]
optimizers = 'adam'

df_results = []

for dropout_rate in dropout_values:
    for num_neurons in neurons_values:
        for batch_size_value in batch_sizes:
            for learning_rate_value in learning_rates:

                print(f"Training model starts for Dropout = {dropout_rate}, Neurons = {num_neurons}, Batch Size = {batch_size_value}, Learning Rate = {learning_rate_value}, Optimizer = {optimizers}")

                set_seeds()
                model = Sequential()
                model.add(SimpleRNN(num_neurons, input_shape=(lags, features), return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(SimpleRNN(num_neurons))
                model.add(Dense(1, activation='sigmoid'))

                optimizer = Adam(learning_rate=learning_rate_value)
                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                history = model.fit(X_train, y_train, 
                                    epochs=150, 
                                    verbose=1,
                                    validation_data=(X_test, y_test),
                                    batch_size=batch_size_value)

                # y_pred
                y_pred = model.predict(X_test, batch_size=None)
                y_pred_binary = (y_pred > 0.5).astype(int)

                accuracy = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                f1 = f1_score(y_test, y_pred_binary)
                auc_roc = roc_auc_score(y_test, y_pred)

                df_results.append({'Dropout      ': dropout_rate,
                                   'Neurons      ': num_neurons,
                                   'Batch Size   ': batch_size_value,
                                   'Learning Rate': learning_rate_value,
                                   'Optimizer    ': optimizers,  
                                   'Accuracy     ': accuracy,
                                   'Precision    ': precision,
                                   'Recall       ': recall,
                                   'F1-Score     ': f1,
                                   'AUC-ROC      ': auc_roc})
                
                #print("Training Loss:", history.history['loss'])
                #print("Training Accuracy:", history.history['accuracy'])
                #print("Validation Loss:", history.history['val_loss'])
                #print("Validation Accuracy:", history.history['val_accuracy'])
                


                print(f"Training model ending for Dropout = {dropout_rate}, Neurons = {num_neurons}, Batch Size = {batch_size_value}, Learning Rate = {learning_rate_value}, Optimizer = {optimizers}")
                print('\n')

                plt.figure(figsize=(12, 6))
                
# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

df_results = pd.DataFrame(df_results)

# Guardar resultados en un archivo Excel
df_results.to_excel('metrics_results.xlsx', index=False)
print("Results saved in: 'metrics_results.xlsx'")

elapsed_time = time.time() - start_time
hours, minutes = divmod(elapsed_time, 3600)
minutes = minutes / 60  # Convertir los minutos restantes a fracción de hora

os.system("afplay /System/Library/Sounds/Ping.aiff")
print(f"Total time taken for the process: {int(hours)} hours, {int(minutes)} minutes")
