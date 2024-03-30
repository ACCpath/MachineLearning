#!/usr/bin/env python3
# coding: utf-8
# executionmodelrisk.py
__version__ = '1.0'
import pathlib
import pandas as pd
import pickle

path_dir = pathlib.Path(__name__).parent.resolve()
# Data processing functions
def data_quality(df):
    temp = df.copy()
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('unknown')
    numeric_columns = temp.select_dtypes('number').columns
    temp[numeric_columns] = temp[numeric_columns].fillna(0)
    temp['vivienda'] = temp['vivienda'].replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp['finalidad'] = temp['finalidad'].replace(['educational', 'reneweable_energy', 'wedding'], 'others')

    return temp

def get_expected_loss(df):
    # Prepare data
    x = data_quality(df)
    
    # LOAD EXECUTION PIPELINES
    
    # Probability Default(PD)
    full_path = path_dir / 'execution_pipe_pd.pickle'
    full_path = full_path.resolve() 
    with open(str(full_path), mode='rb') as file:
        execution_pipe_pd = pickle.load(file)

    # Exposure at Default (EAD)
    full_path = path_dir / 'execution_pipe_ead.pickle'
    full_path = full_path.resolve()
    with open(str(full_path), mode='rb') as file:
        execution_pipe_ead = pickle.load(file)

    # Loss Given Default (LGD)
    full_path = path_dir / 'execution_pipe_lgd.pickle'
    full_path = full_path.resolve()
    with open(str(full_path), mode='rb') as file:
        execution_pipe_lgd = pickle.load(file)
    
    # Execution
    scoring_pd = execution_pipe_pd.predict_proba(x)[:, 1]
    ead = execution_pipe_ead.predict(x)
    lgd = execution_pipe_lgd.predict(x)

    # Expected Loss(EL)
    principal = x.principal
    EL = pd.DataFrame({'principal': principal, 'pd': scoring_pd, 'ead': ead, 'lgd': lgd})
    EL['expected_loss'] = round(EL.pd * EL.principal * EL.ead * EL.lgd, 2)
    
    return EL
