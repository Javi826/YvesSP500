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
from paths.paths import file_df_data,folder_csv,path_file_csv,results_path,path_tra_val_results,file_tra_val_results, path_base,folder_tra_val_results
from columns.columns import columns_csv_yahoo,columns_clean_order
from functions.def_functions import set_seeds, class_weight,plots_histograms,plot_loss, plot_accu
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from modules.mod_pipeline import mod_pipeline

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from keras.regularizers import l2
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
df_clean = mod_dtset_clean(df_data,start_date,endin_date)

#CALL PREPROCESSING
#------------------------------------------------------------------------------
prepro_start_date = '2000-01-01'
prepro_endin_date = '2019-12-31'
lags_range        = [20]

for lags in lags_range:
    
    df_preprocessing = mod_preprocessing(df_clean,prepro_start_date,prepro_endin_date,lags)
    #print(df_preprocessing)
    
    #CALL PIPELINE
    #------------------------------------------------------------------------------
    n_features = 1
    endin_data_train  = initn_data_valid  = ['2018-01-01']
    endin_data_valid  = '2018-12-31'
    
    print(f"Starts Processing for lags = {lags} and initn_data_valid = {initn_data_valid}")
    print('\n')
    
    X_train = mod_pipeline(df_preprocessing, endin_data_train, endin_data_valid,lags, n_features, 'X_train')
    y_train = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'y_train')
    X_valid = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid')
    y_valid = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'y_valid')

    df_results = []
            
    #LOOPs 2
    #------------------------------------------------------------------------------
    dropout_range = [0.1]
    neurons_range = [30]
    batch_s_range = [32]
    le_rate_range = [0.001]
    optimizers    = 'adam'
    #optimizers_to_try = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
    
    for dropout in dropout_range:
        for n_neurons in neurons_range:
            for batch_s in batch_s_range:
                for le_rate in le_rate_range:
                    print(f"Training model starts for Dropout = {dropout}, Neurons = {n_neurons}, Batch Size = {batch_s}, Learning Rate = {le_rate}, Optimizer = {optimizers}")
    
                    set_seeds()
                    model = Sequential()
                    model.add(SimpleRNN(n_neurons, input_shape=(lags, n_features), return_sequences=True))
                    model.add(Dropout(dropout))
                    model.add(SimpleRNN(n_neurons, kernel_regularizer=l2(0.0001)))
                    model.add(Dense(1, activation='sigmoid'))
    
                    optimizer = Adam(learning_rate=le_rate)
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
                    file_model_name = f'version01.keras'
                    path_keras = (results_path / file_model_name).as_posix()
                    
                    checkpointer = ModelCheckpoint(filepath=path_keras, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
                    early_stopping = EarlyStopping(monitor='loss', patience=15, verbose=1, restore_best_weights=True)
                    
                    history = model.fit(X_train, y_train, 
                                        epochs=30, 
                                        verbose=0,
                                        batch_size=batch_s,
                                        validation_data=(X_valid, y_valid),
                                        callbacks=[checkpointer, early_stopping])
                    
                    accuracy_history = pd.DataFrame(history.history)
                    #print(accuracy_history)
                    accuracy_history.index += 1
                    
                    # Encontrar la mejor precisión de validación y la época correspondiente
                    best_accur = accuracy_history['val_accuracy'].max()
                    best_epoch = accuracy_history['val_accuracy'].idxmax()
                    
                    # Imprimir la mejor precisión de validación y la época correspondiente
                    print(f"Best val_accuracy: {best_accur:.4f}")
                    print(f"Best epoch: {best_epoch}")
                 
                    # Training metrics
                    train_loss = history.history['loss'][-1]
                    train_accu = history.history['accuracy'][-1]
                    valid_loss = history.history['val_loss'][-1]
                    valid_accu = history.history['val_accuracy'][-1]
    
                    df_results.append({
                        'Lags': lags,
                        'Cutoff Date': initn_data_valid,
                        'Dropout': dropout,
                        'Neurons': n_neurons,
                        'Batch Size': batch_s,
                        'Learning Rate': le_rate,
                        'Optimizer': optimizers,
                        'Train Loss': train_loss,
                        'Val Loss': valid_loss,
                        'Train Accu': train_accu,
                        'Val Accu': valid_accu,
                        'Best val_accuracy': best_accur,
                        'Best epoch': best_epoch
                    })
    
                    print(f"Training model ending for Dropout = {dropout}, Neurons = {n_neurons}, Batch Size = {batch_s}, Learning Rate = {le_rate}, Optimizer = {optimizers}")
                    print('\n')
    
                    
                    #plot_loss(history)
                    plot_accu(history)
    
    print(f"Ending Processing ending for lags = {lags} and initn_data_valid = {initn_data_valid}")
    print('\n')
            
    df_tra_val_results = pd.DataFrame(df_results)
    excel_file_path = os.path.join(path_base, folder_tra_val_results, f"df_tra_val_all.xlsx")
    df_tra_val_results.to_excel(excel_file_path, index=False)
    print("All Training results saved in: 'tra_val_results/df_tra_val_results.xlsx'")

elapsed_time = time.time() - start_time
hours, minutes = divmod(elapsed_time, 3600)
minutes = minutes / 60  

os.system("afplay /System/Library/Sounds/Ping.aiff")
print(f"Total time taken for the process: {int(hours)} hours, {int(minutes)} minutes")
