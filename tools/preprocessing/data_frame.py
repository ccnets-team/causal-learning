'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import numpy as np
import pandas as pd
from typing import Tuple, List, Tuple
from tools.preprocessing.utils import get_columns, generate_description, remove_process_prefix
from tools.preprocessing.utils import calculate_num_classes, convert_to_indices, remove_columns, display_statistics
from tools.preprocessing.scaler import auto_scale_columns
from tools.preprocessing.encode import one_hot_encode_columns, encode_label_columns
from tools.preprocessing.utils import PROCESSED_PREFIX 

def auto_encode_cyclical_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Preprocesses specified cyclical columns in the DataFrame by applying sine and cosine transformations.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (List[str]): List of column names to be processed.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns processed.
    """
    
    processed_columns = {}
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        
        # Apply sine and cosine transformations
        df[PROCESSED_PREFIX + col + '_sin'] = np.sin(2 * np.pi * (df[col] - min_val) / range_val)
        df[PROCESSED_PREFIX + col + '_cos'] = np.cos(2 * np.pi * (df[col] - min_val) / range_val)
        df.drop(col, axis=1, inplace=True)
        processed_columns[col] = [PROCESSED_PREFIX + col + '_sin', PROCESSED_PREFIX + col + '_cos']
    
    return df, processed_columns

def process_time_column(df, actual_time_col, prefix):
    datetime = pd.to_datetime(df[actual_time_col], errors='coerce', format='%H:%M:%S', exact=False)
    
    if datetime.notnull().all() and isinstance(datetime.iloc[0], pd.Timestamp):
        if datetime.dt.date.notnull().all():
            if 'date' not in df.columns:
                df['date'] = datetime.dt.date
            datetime = datetime.dt.time

        df[prefix + 'hours'] = datetime.apply(lambda x: x.hour if pd.notnull(x) else np.nan)
        df[prefix + 'minutes'] = datetime.apply(lambda x: x.minute if pd.notnull(x) else np.nan)
        df[prefix + 'seconds'] = datetime.apply(lambda x: x.second if pd.notnull(x) else np.nan)
        
        df[prefix + 'total_seconds'] = df[prefix + 'hours'] * 3600 + df[prefix + 'minutes'] * 60 + df[prefix + 'seconds']
        df[prefix + 'time_scaled'] = 2 * (df[prefix + 'total_seconds'] / 86400) - 1
        
        df.drop([actual_time_col, prefix + 'hours', prefix + 'minutes', prefix + 'seconds', prefix + 'total_seconds'], axis=1, inplace=True)

        return [prefix + 'time_scaled']
    return []

def process_date_column(df, actual_date_col, prefix):
    datetime = pd.to_datetime(df[actual_date_col], errors='coerce')
    
    if datetime.notnull().all():
        df['day_of_year'] = datetime.dt.dayofyear
        
        df[prefix + 'day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df[prefix + 'day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        if 'month' in df.columns:
            df.drop(['month'], axis=1, inplace=True)

        df.drop([actual_date_col, 'day_of_year'], axis=1, inplace=True)
        return [prefix + 'day_of_year_sin', prefix + 'day_of_year_cos']
    return []

def process_month_column(df, actual_month_col, prefix):
    df[actual_month_col] = pd.to_numeric(df[actual_month_col], errors='coerce')
    
    if df[actual_month_col].between(1, 12).all():
        df[prefix + 'month_sin'] = np.sin(2 * np.pi * df[actual_month_col] / 12)
        df[prefix + 'month_cos'] = np.cos(2 * np.pi * df[actual_month_col] / 12)
        
        df.drop([actual_month_col], axis=1, inplace=True)
        return [prefix + 'month_sin', prefix + 'month_cos']
    return []


def auto_encode_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically encodes datetime-related columns in the DataFrame.
    - Converts the 'date' column to sine and cosine transformations of the day of the year.
    - Encodes the 'time' column linearly from -1 to 1.
    - Handles cases where the 'time' column contains date information.
    - Considers both lowercase and capitalized versions of 'time' and 'month'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'date' and/or 'time' columns.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'date', 'time', and 'month' columns processed.
    """
    
    prefix = PROCESSED_PREFIX

    # Define possible column names
    time_cols = ['time', 'Time']
    date_cols = ['date', 'Date']
    month_cols = ['month', 'Month']

    processed_columns = []

    # Find the actual columns present in the DataFrame
    actual_time_col = next((col for col in time_cols if col in df.columns), None)
    actual_date_col = next((col for col in date_cols if col in df.columns), None)
    actual_month_col = next((col for col in month_cols if col in df.columns), None)

    # Process columns
    if actual_time_col:
        processed_columns += process_time_column(df, actual_time_col, prefix)
        
    if actual_date_col:
        processed_columns += process_date_column(df, actual_date_col, prefix)
        
    if actual_month_col:
        processed_columns += process_month_column(df, actual_month_col, prefix)

    encoded_columns = convert_to_indices(df, processed_columns)
    return df, encoded_columns        

def auto_preprocess_dataframe(df: pd.DataFrame, target_columns, drop_columns = None, one_hot_columns = None) -> Tuple[pd.DataFrame, dict]:
    """
    Automatically preprocesses the DataFrame by encoding, scaling, and handling target columns.
    """

    # Convert columns to DataFrame indices
    target_columns, drop_columns, one_hot_columns = \
        convert_to_indices(df, target_columns, drop_columns, one_hot_columns)

    # Drop unwanted columns
    df = remove_columns(df, drop_columns)
    
    # Split DataFrame into feature columns and target columns
    df_x, df_y = df.drop(columns=target_columns), df[target_columns]
    
    df_x, encoded_datatime_columns = auto_encode_datetime_columns(df_x)
    
    # Encode non-target columns
    df_x, encoded_one_hot_columns = one_hot_encode_columns(df_x, one_hot_columns)
    
    # Scale non-target columns
    df_x, scaler_description = auto_scale_columns(df_x)
    
    # Convert all features to float type
    df_x = df_x.astype(float)
    
    # Calculate the number of features and the number of classes in the target columns
    num_features = df_x.shape[1]
    num_classes = calculate_num_classes(df_y)
    
    # Encode target columns
    df_y, encoded_target_columns = encode_label_columns(df_y)

    # Concatenate processed feature columns and target columns
    df = pd.concat([df_x, df_y], axis=1)
    
    # Remove internal process tags from column names
    df = remove_process_prefix(df)

    # combine encoded columns
    encoded_columns = encoded_datatime_columns.union(encoded_one_hot_columns).union(encoded_target_columns)
    
    # Generate description dictionary
    description = generate_description(num_features=num_features, num_classes=num_classes, 
                                       encoded_columns=encoded_columns, scalers=scaler_description)
    
    # Display DataFrame statistics
    display_statistics(df)

    # Return processed DataFrame and description
    return df, description
