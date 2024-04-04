#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:06:19 2023
@author: jlahoz
"""

import os
from pathlib import Path

## Directorio para almacenar archivos CSV
path_base = "/Users/javi/Desktop/ML/YvesSP500"

file_df_data           = "sp500_data.csv"
folder_csv             = "inputs/historicyh"
path_file_csv          = os.path.join(path_base, folder_csv, file_df_data)

file_df_clean          = "df_clean.csv"
folder_df_clean        = "inputs/dtset_clean"
path_df_clean          = os.path.join(path_base, folder_df_clean, file_df_clean)

file_summary_stats     = 'df_summary_stats'
folder_summary_stats   = "outputs/summary_stats"
path_summary_stats     = os.path.join(path_base, folder_summary_stats, file_summary_stats)

file_preprocessing     = 'df_preprocessing.xlsx'
folder_preprocessing   = "inputs/preprocessing"
path_preprocessing     = os.path.join(path_base, folder_preprocessing, file_preprocessing)

file_tra_val_results   = 'df_tra_val_results.xlsx'
folder_tra_val_results = "results/tra_val_results"
path_tra_val_results   = os.path.join(path_base, folder_tra_val_results, file_tra_val_results)

file_tests_results     = 'df_tests_results.xlsx'
folder_tests_results   = "results/tests_results"
path_tests_results     = os.path.join(path_base, folder_tests_results, file_tests_results)

folder_tf_serving = "tf_serving"
tf_serving_path = os.path.join(path_base, folder_tf_serving)

results_path = Path('/Users/javi/Desktop/ML/YvesSP500/keras')






