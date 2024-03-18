#!/usr/bin/env python3
# coding: utf-8
# executionretail.py
__version__ = '1.0'

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from functionsretail import run_recursive_forecast

# Load data
project_path = '../'
file_name = 'data_for_production.csv'
full_path = os.path.join(project_path, 'data/validation', file_name)
df = pd.read_csv(full_path, sep=';', parse_dates=['date'], index_col='date')

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

#Launch prediction
forecast = run_recursive_forecast(df, project_path)
forecast.sort_values(by=['store_id', 'item_id'])

