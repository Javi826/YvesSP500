#FUNTCIONS

"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

from columns.columns import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.layers import SimpleRNN, LSTM, Dense,Dropout
from keras.models import Sequential

lags = 5 

def day_week(df_clean):
       
    # column with dates
    date_column = 'date' 
    # ensuring date_column with date format
    df_clean[date_column] = pd.to_datetime(df_clean[date_column])
    # add column day_week + from label to number
    df_clean['day_week'] = df_clean[date_column].dt.dayofweek  
    
    return df_clean


def add_index_column(df_clean):
    
    # add index
    df_clean.insert(0, 'index_id', range(1, len(df_clean) + 1))
    df_clean['index_id'] = df_clean['index_id'].apply(lambda x: f'{x:05d}')
    
    
    return df_clean

def date_anio(df_clean):
    
    df_clean['date'] = pd.to_datetime(df_clean['date'])    
    # Extract year
    df_clean['date_anio'] = df_clean['date'].dt.year.astype(str).str[:4]
    
    return df_clean


def sort_columns(df_clean):

    desired_column_order = columns_clean_order
    # Ensuure columns in dataframe
    missing_columns = set(desired_column_order) - set(df_clean.columns)
    if missing_columns:
        raise ValueError(f"following columns no in DataFrame: {', '.join(missing_columns)}")

    # Sort columns
    df_clean = df_clean[desired_column_order]
    
    return df_clean

def rounding_data(df_clean):

    columns_to_round = ['open', 'high', 'low', 'close', 'adj_close']
    # format float
    df_clean[columns_to_round] = df_clean[columns_to_round].astype(float)
    df_clean['day_week'] = df_clean['day_week'].astype(int)
    #format rounding 
    for column in columns_to_round:
      if column in df_clean.columns:
          df_clean[column] = df_clean[column].round(4)
            
    return df_clean

def filter_data_by_date_range(df, filter_start_date, filter_endin_date):
        
    return df[(df['date'] >= filter_start_date) & (df['date'] <= filter_endin_date)]

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def class_weight(df_preprocessing):
    
    c0, c1 = np.bincount(df_preprocessing['direction'])
    w0 = (1/c0) * (len(df_preprocessing)) / 2
    w1 = (1/c1) * (len(df_preprocessing)) / 2
    return {0: w0, 1:w1}


def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accu(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def df_plots(x, y, x_label, y_label,plot_style):
    
    plt.figure(figsize=(10, 6))
    
    if plot_style == "lines":
        plt.plot(x, y, label=f'{x_label} vs {y_label}')  # Use a line plot
    elif plot_style == "points":
        plt.scatter(x, y, label=f'{x_label} vs {y_label}', marker='o')  # Use a scatter plot with markers
    else:
        raise ValueError("Invalid plot_style. Use 'lines' or 'points'.")

    plt.title(f'{x_label} vs {y_label} Plot')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()
            
def plots_histograms(dataframe, columns_of_interest):
    bins = 30
    figsize = (15, 5)
    
    # plot
    fig, axes = plt.subplots(nrows=1, ncols=len(columns_of_interest), figsize=figsize)
    
    # Columns
    for i, column in enumerate(columns_of_interest):
        axes[i].hist(dataframe[column], bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Histogram de {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('frequency')
    
    # design
    plt.tight_layout()
    plt.show()

