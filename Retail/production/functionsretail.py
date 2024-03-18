#!/usr/bin/env python3
# coding: utf-8
# functionsretail.py
"""functionsRetail is a module that contains functions to process historical sales data to train or execute predictive models at the store-product level of a large distributor in the food sector.
"""
__version__ = '1.0'

import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


def impute_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)
    return df

def apply_data_quality(x):
    temp = x.astype({'month': 'category', 'wday': 'category'})
    vars_impute = ['event_name_1']
    temp[vars_impute] = temp[vars_impute].fillna('No_event')
    temp = (temp.groupby('item_id', group_keys=False)
            .apply(lambda x: impute_mode(x, 'sell_price')))
    
    return temp


# In[ ]:


def determine_stock_out(sales, n=5):
    zero_sales = pd.Series(np.where(sales==0, 1, 0))
    num_zeros = zero_sales.rolling(n).sum()
    return np.where(num_zeros==n, 1, 0)

def generate_lags(df, variable, num_lags=7):
    """Generates lagged versions of a specific variable in a DataFrame.
    """
    lags = pd.DataFrame({
        f'{variable}_lag_{lag}': df[variable].shift(lag)
        for lag in range(1, num_lags + 1)
    })
    return lags

def calculate_mobile_window(df, variable, window_function, num_periods=7):
    """Calculates the rolling statistics (min, mean, max) for a specific variable
    using historical data shifted down in a period for each specified number of periods.
    """
    window_name = window_function.__name__
    return pd.DataFrame({
        f'{variable}_{window_name}_{roll}': df[variable].shift(1).rolling(roll).apply(window_function)
        for roll in range(2, num_periods + 1)
    })

def generate_variables(x):
    x = x.sort_values(['store_id', 'item_id', 'date'])
    #Variables Intermittent demand
    stock_out_windows = [3, 7, 15]
    for window_size in stock_out_windows:
        x[f'stock_out_{window_size}'] = (
            x.groupby(['store_id', 'item_id'])
            .sales
            .transform(lambda x: determine_stock_out(x, window_size))
        )
    
    # Lags
    lag_periods = {
        'sell_price': 7, 'stock_out_3': 1, 'stock_out_7': 1,
        'stock_out_15': 1, 'sales': 15
    }
    lags_dfs = []
    for variable, lag_period in lag_periods.items():
        lag_df = (
            x.groupby(['store_id', 'item_id'])
            .apply(lambda x: generate_lags(x, variable, lag_period))
            .reset_index()
            .set_index('date')
        )
        lags_dfs.append(lag_df)
    lags_dfs = pd.concat(lags_dfs, axis=1)

    # Mobile windows
    window_functions = [np.min, np.mean, np.max]
    mobile_dfs = []

    for window_function in window_functions:
        mobile_df = (
            x.groupby(['store_id', 'item_id'])
            .apply(lambda x: calculate_mobile_window(x, 'sales', window_function, 15))
            .reset_index()
            .set_index('date')
        )
        mobile_dfs.append(mobile_df)
    mobile_dfs = pd.concat(mobile_dfs, axis=1)
    # Holidays variable
    x['festive'] = np.where(x['event_name_1'] == 'No_event', 0, 1)
    
    # Combine DataFrames    
    x_combined = pd.concat([x, lags_dfs, mobile_dfs], axis=1)

    # Remove duplicate columns
    x_combined = x_combined.loc[:,~x_combined.columns.duplicated()]
    
    # Drop NaN values
    x_combined.dropna(inplace=True)
    
    # Remove unnecessary columns
    vars_delete = ['sell_price', 'stock_out_3', 'stock_out_7', 'stock_out_15']
    x_combined.drop(columns=vars_delete, inplace=True)
    
    x_combined.insert(loc=0, column='productf',
                      value=x_combined.store_id + '_' + x_combined.item_id)   
    x_combined = x_combined.drop(columns=['store_id', 'item_id'])
    
    return x_combined


# In[ ]:


def transform_variables(x, y=None, way='train', project_path='route/to/project'):
    x.reset_index(inplace=True)

    # Manage encoders and categorical variables
    var_cat = ['month', 'wday', 'weekday', 'event_name_1']
    encoders = {
        'ohe': {'name': 'ohe_retail.pickle', 'vars': var_cat},
        'te': {'name': 'te_retail.pickle', 'vars': var_cat}
    }

    for encoder_type, config in encoders.items():
        path = os.path.join(project_path, 'models', config['name'])
        if encoder_type == 'ohe' :
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            encoder = TargetEncoder(min_samples_leaf=100, return_df=False)
        
        if way == 'train':
            if encoder_type == 'te':
                y.reset_index(inplace=True, drop=True)
                y = y.loc[y.index.isin(x.index)]
            encoder.fit(x[config['vars']], y=y if encoder_type == 'te' else None)
            
            # Save the trained encoder
            with open(path, mode='wb') as file:
                pickle.dump(encoder, file)
        else:
            # Load saved encoder
            with open(path, mode='rb') as file:
                encoder = pickle.load(file)
        
        # Transform variables
        encoded_data = encoder.transform(x[config['vars']])
        
        # Get the names of the encoded columns
        if encoder_type == 'te':
            col_names = [f'{var}_encoded' for var in config['vars']]
        else:
            # Get the names of the binary columns generated by OneHotEncoder
            col_names = encoder.get_feature_names_out(input_features=config['vars'])
        
        # Update "x" with hardcoded variables
        x_encoded = pd.DataFrame(
            encoded_data,
            columns=col_names,
            index = x.index
        )
        x = pd.concat([x, x_encoded], axis=1)
    
    x.drop(columns=var_cat, inplace=True) 
    x.set_index('date', inplace=True)
    
    return x


# In[ ]:


def preselect_variables(x, y, position_variable_limit=80):
    x = x.drop(columns='productf').reset_index(drop=True)
    
    # Align y with the filtered indices of x
    y = y.loc[y.index.isin(x.index)]
    
    # Calculate mutual information
    mutual_info = mutual_info_regression(x, y)
    
    # Create DataFrame for ranking mutual information
    df_ranking = pd.DataFrame({'variable': x.columns, 'importance_mi': mutual_info}) \
                    .sort_values(by='importance_mi', ascending=False) \
                    .reset_index(drop=True)
    
    # Add ranking based on mutual information
    df_ranking['ranking_mi'] = df_ranking.index
    
    # Select top variables based on mutual information
    selected_vars = df_ranking.iloc[:position_variable_limit]['variable']
    # Add  Holidays
    selected_vars = selected_vars.to_list() + ['festive']
    # Select only the top variables in x
    x_mi = x[selected_vars].copy()
    
    return x_mi


# In[ ]:


def model_product(x_product, y_product):
    
    # Select predictor variables
    select_vars = x_product.columns[2:].to_list()
    
    # Define cross validation settings
    time_cv = TimeSeriesSplit(5, test_size=8)
    
    # Define the pipeline estimator and the hyperparameter grid
    pipe = Pipeline([('algorithm', HistGradientBoostingRegressor)])
    grid = [{'algorithm': [HistGradientBoostingRegressor()]}]
    
    # Perform random hyperparameter search
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=1,
        cv=time_cv,
        scoring='neg_mean_absolute_error', 
        verbose=0,
        n_jobs=-1
    )
    # Train the final model
    final_model = random_search.fit(x_product[select_vars], y_product).best_estimator_
    
    return final_model


# In[ ]:


def launch_training(df, project_path):
    """Train models for each product in the dataframe and save them.
    """
    list_models = []
    for product in df['productf'].unique():
        df_product = df[df['productf'] == product].copy()
        x = df_product.drop(columns='sales').copy()
        y = df_product['sales'].copy()
        x = transform_variables(x=x, y=y, project_path=project_path)
        x = preselect_variables(x, y)
        model = model_product(x, y)
        list_models.append((product, model))
    file_models = 'list_models_retail.pickle'
    path_models = os.path.join(project_path, 'models', file_models)
    with open(path_models, mode='wb') as file:
        pickle.dump(list_models, file)
    


# In[ ]:


def launch_execution(df, project_path):
    """Run the models trained with the provided data and generate predictions
    for the corresponding day.
    """
    # Load the trained models
    file_models = 'list_models_retail.pickle'
    path_models = os.path.join(project_path, 'models', file_models)
    with open(path_models, mode='rb') as file:
        list_models = pickle.load(file)
    
    prediction_df = pd.DataFrame(columns=['date', 'productf', 'sales', 'prediction'])
    
    # Iterate over each product and its respective model
    for product, model in list_models:
        
        # Filter the dataframe by product
        df_product = df[df['productf'] == product].copy()
        
        # Transform input variables
        x = transform_variables(x=df_product.drop(columns='sales'), way='execution',
                                project_path=project_path)
        # Select variables
        x = x[model[0].feature_names_in_]
        
        # Make predictions and store the results
        prediction_df = pd.concat([prediction_df, pd.DataFrame({
            'date': df_product.index.values,
            'productf': product,
            'sales': df_product['sales'],
            'prediction': model.predict(x).astype(int)
            
        })])
    # Update negative predictions
    prediction_df.loc[(prediction_df['prediction'] < 0), 'prediction'] = 0
    # Return only predictions corresponding to the minimum date
    return prediction_df.loc[prediction_df.index == prediction_df.index.min()]


# In[ ]:


def run_recursive_forecast(x, project_path):
    """Calculate sales prediction for the next 8 days using an iterative approach.
    Args:
        x(DataFrame): Sales history with file structure "data_for_production.csv"
        located in the directory: /data/validation
    Returns:
        DataFrame updated with sales predictions
    """
    for _ in range(8):
        step1_df = apply_data_quality(x.copy())
        step2_df = generate_variables(step1_df)
        
        predictions_df =  launch_execution(step2_df, project_path)
        predictions_df['store_id'] = predictions_df['productf'].str[:4]
        predictions_df['item_id'] = predictions_df['productf'].str[5:]

        sales_update_condition = (
            x.index.isin(predictions_df['date'])
            & x['store_id'].isin(predictions_df['store_id'])
            & x['item_id'].isin(predictions_df['item_id'])
        )
        x.loc[sales_update_condition, 'sales'] = predictions_df['prediction']
                                                              
        x = x[x.index != x.index.min()]
        
    return x

