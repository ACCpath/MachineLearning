#!/usr/bin/env python3
# coding: utf-8
# trainingretail.py
__version__ = '1.0'

import os
import pandas as pd
from functionsretail import apply_data_quality, generate_variables, launch_training

import warnings
warnings.filterwarnings("ignore")

# Load data
project_path = '../'
file_name = 'work.csv'
full_path = os.path.join(project_path, 'data', file_name)
df = pd.read_csv(full_path, sep=',', parse_dates=['date'], index_col='date')

#Select variables
final_vars = [
    'store_id',
    'item_id',
    'event_name_1',
    'month',
    'sell_price',                      
    'wday',
    'weekday',
    'sales']
df = df[final_vars]

# Train models
step1_df = apply_data_quality(df)
step2_df = generate_variables(step1_df)
launch_training(step2_df, project_path)
