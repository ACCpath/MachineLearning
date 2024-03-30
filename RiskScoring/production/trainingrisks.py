#!/usr/bin/env python3
# coding: utf-8
# trainingrisks.py
__version__ = '1.0'

import os
import numpy as np
import pandas as pd
import pickle

# Load data
project_path = '../'
file_name = 'prestamos.csv'
full_path = os.path.join(project_path, 'data/original', file_name)
df = pd.read_csv(full_path, index_col='id_cliente')

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
    'estado',
    'imp_amortizado',
    'imp_recuperado'
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

def generate_variables_pd(df):
    """Generate PD (Probability of Default) related variables from a given DataFrame.
    Returns:
    tuple: A pair of DataFrames, the first one containing the predictor variables
    and the second one containing the PD target.
    """
    temp = df.copy()
    default = ['Charged Off', 'Does not meet the credit policy. Status:Charged Off', 'Default']
    temp['target_pd'] = np.where(temp.estado.isin(default), 1, 0)
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado'], inplace=True)
    
    return temp.iloc[:,:-1], temp.iloc[:,-1]

def generate_variables_ead(df):
    """Generate EAD (Exposure at Default) related variables from a given DataFrame.
    Returns:
    tuple: A pair of DataFrames, the first one containing the predictor variables
    and the second one containing the EAD target.
    """
    temp = df.copy()
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['target_ead'] = temp.pendiente / temp.principal
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    
    return temp.iloc[:, :-1], temp.iloc[:, -1]

def generate_variables_lgd(df):
    """Generate Loss Given Default (LGD) related variables from a given DataFrame.
    Returns:
    tuple: A pair of DataFrames, the first one containing the predictor variables
    and the second one containing the LGD target.
    """
    temp = df.copy()
    temp['pendiente'] = temp['principal'] - temp['imp_amortizado']
    temp['target_lgd'] = 1 - (temp.imp_recuperado / temp.pendiente).fillna(0)
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    
    return temp.iloc[:, :-1], temp.iloc[:, -1]

# Prepare dataset
x_pd, y_pd = generate_variables_pd(data_quality(df))
x_ead, y_ead = generate_variables_ead(data_quality(df))
x_lgd, y_lgd = generate_variables_lgd(data_quality(df))

# Load training pipeline
# Probability Default(PD)
full_path = os.path.join(project_path, 'models', 'training_pipe_pd.pickle')
with open(full_path, mode='rb') as file:
    training_pipe_pd = pickle.load(file)

# Exposure at Default (EAD)
full_path = os.path.join(project_path, 'models', 'training_pipe_ead.pickle')
with open(full_path, mode='rb') as file:
    training_pipe_ead = pickle.load(file)

# Loss Given Default (LGD)
full_path = os.path.join(project_path, 'models', 'training_pipe_lgd.pickle')
with open(full_path, mode='rb') as file:
    training_pipe_lgd = pickle.load(file)
    
# Training
execution_pipe_pd = training_pipe_pd.fit(x_pd, y_pd)
execution_pipe_ead = training_pipe_ead.fit(x_ead, y_ead)
execution_pipe_lgd = training_pipe_lgd.fit(x_lgd, y_lgd)

# Save execution pipeline
full_path = os.path.join(project_path, 'models', 'execution_pipe_pd.pickle')
with open(full_path, mode='wb') as file:
    pickle.dump(execution_pipe_pd, file)

full_path = os.path.join(project_path, 'models', 'execution_pipe_ead.pickle')
with open(full_path, mode='wb') as file:
    pickle.dump(execution_pipe_ead, file)

full_path = os.path.join(project_path, 'models', 'execution_pipe_lgd.pickle')
with open(full_path, mode='wb') as file:
    pickle.dump(execution_pipe_lgd, file)
