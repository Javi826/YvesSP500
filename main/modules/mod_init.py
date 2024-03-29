#MODULE INIT
"""
Created on Mon Nov  8 22:54:48 2023
@author: jlahoz
"""

import warnings
import time
import os
import pandas as pd


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# IGNORE WARNINGS
warnings.filterwarnings("ignore")

#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
