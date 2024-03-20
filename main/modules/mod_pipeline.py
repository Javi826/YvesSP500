#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
#from functions.def_functions import *
from paths.paths import path_base,folder_preprocessing
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functions.def_functions import filter_data_by_date_range, df_plots
from paths.paths import path_base, folder_preprocessing

def mod_pipeline(df_preprocessing, initn_date_range, endin_date_range, lags, n_features, data_type):
    
    X_train, y_train, X_valid, y_valid = None, None, None, None
    
    for cutoff_date in initn_date_range:
        #print(f"Pipeline for {data_type}: start with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")
        
        df_columns = ['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction']
        df_date_lag_dir = df_preprocessing[df_columns].copy()
                  
        #DATA SPLIT
        #------------------------------------------------------------------------------  
        
        train_data = df_date_lag_dir[df_date_lag_dir['date'] <= cutoff_date]
        valid_data = df_date_lag_dir[(df_date_lag_dir['date'] > cutoff_date) & (df_date_lag_dir['date'] <= endin_date_range)]
        tests_data = df_date_lag_dir[(df_date_lag_dir['date'] > cutoff_date) & (df_date_lag_dir['date'] <= endin_date_range)]
        #print(train_data)
        #print(valid_data)
        #print(tests_data)
        
        lag_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('lag')]
        #print(lag_columns_selected)
        
        #X_TRAIN & y_train | NORMALIZATION + RESHAPE
        #------------------------------------------------------------------------------
        
        if data_type == 'X_train':
            X_data = train_data[lag_columns_selected]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            X_scaled = pd.DataFrame(X_scaled, columns=lag_columns_selected)
            X_reshaped = X_scaled.values.reshape(-1, lags, n_features)
            X_train = X_reshaped
            
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")

            return X_train
            
        elif data_type == 'y_train':
            y_train = train_data['direction']
            
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")

            return y_train
            
        #X_VALID & y_valid | NORMALIZATION + RESHAPE
        #------------------------------------------------------------------------------
        
        elif data_type == 'X_valid':
            X_data = valid_data[lag_columns_selected]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            X_scaled = pd.DataFrame(X_scaled, columns=lag_columns_selected)
            X_reshaped = X_scaled.values.reshape(-1, lags, n_features)
            X_valid = X_reshaped
            
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")

            return X_valid
        
        elif data_type == 'y_valid':
            y_valid = valid_data['direction']
            
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")

            return y_valid
        
        #X_TESTS & y_tests | NORMALIZATION + RESHAPE
        #------------------------------------------------------------------------------        
        elif data_type == 'X_tests':
            X_data = tests_data[lag_columns_selected]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            X_scaled = pd.DataFrame(X_scaled, columns=lag_columns_selected)
            X_reshaped = X_scaled.values.reshape(-1, lags, n_features)
            X_tests = X_reshaped
            
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")

            return X_tests
        
        elif data_type == 'y_tests':
            y_tests = valid_data['direction']
            #print(f"Pipeline for {data_type}: endin with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")
            
            return y_tests
            