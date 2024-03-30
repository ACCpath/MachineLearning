#!/usr/bin/env python3
# coding: utf-8
# executionrisks.py
__version__ = '1.0'

import os
import pandas as pd
import pickle

# Load data
project_path = '../'
file_name = 'validation.csv'
full_path = os.path.join(project_path, 'data/validation', file_name)
df = pd.read_csv(full_path, index_col='id_cliente').drop(columns='Unnamed: 0')

# Select variables
final_variables = [
    'ingresos_verificados',
    'vivienda',
    'finalidad',
    'num_cuotas',
    'antigüedad_empleo',
    'rating',
    'ingresos',
    'dti',
    'num_lineas_credito',
    'porc_uso_revolving',
    'principal',
    'tipo_interes',
    'imp_cuota',
    'num_derogatorios',
]
df.drop_duplicates(inplace=True)
to_eliminate = df.loc[df.ingresos > 300_000].index.values
df = df[~df.index.isin(to_eliminate)]
df = df[final_variables]

# Data processing functions
def data_quality(df):
    temp = df.copy()
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('unknown')
    numeric_columns = temp.select_dtypes('number').columns
    temp[numeric_columns] = temp[numeric_columns].fillna(0)
    temp['vivienda'] = temp['vivienda'].replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp['finalidad'] = temp['finalidad'].replace(['educational', 'reneweable_energy', 'wedding'], 'others')

    return temp

# Prepare dataset
x = data_quality(df)

# Load execution pipeline
# Probability Default(PD)
full_path = os.path.join(project_path, 'models', 'execution_pipe_pd.pickle')
with open(full_path, mode='rb') as file:
    execution_pipe_pd = pickle.load(file)

# Exposure at Default (EAD)
full_path = os.path.join(project_path, 'models', 'execution_pipe_ead.pickle')
with open(full_path, mode='rb') as file:
    execution_pipe_ead = pickle.load(file)

# Loss Given Default (LGD)
full_path = os.path.join(project_path, 'models', 'execution_pipe_lgd.pickle')
with open(full_path, mode='rb') as file:
    execution_pipe_lgd = pickle.load(file)
    
# Execution
scoring_pd = execution_pipe_pd.predict_proba(x)[:, 1]
ead = execution_pipe_ead.predict(x)
lgd = execution_pipe_lgd.predict(x)

# Expected Loss(EL)
principal = x.principal
EL = pd.DataFrame({
    'principal': principal,
    'pd': scoring_pd,
    'ead': ead,
    'lgd': lgd
})
EL['expected_loss'] = round(EL.pd * EL.principal * EL.ead * EL.lgd, 2)