'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import numpy as np
import pandas as pd
from typing import List
from preprocessing.utils import convert_to_indices
from preprocessing.utils import PROCESSED_PREFIX 

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

def parse_datetime_column(value, date_formats, time_formats):
    # Try time formats first
    for fmt in time_formats:
        try:
            parsed_time = pd.to_datetime(value, format=fmt, errors='raise')
            return parsed_time.time() if isinstance(parsed_time, pd.Timestamp) else parsed_time
        except (ValueError, TypeError):
            continue
    # Try date formats
    for fmt in date_formats:
        try:
            return pd.to_datetime(value, format=fmt, errors='raise')
        except (ValueError, TypeError):
            continue
    return pd.NaT  # If all formats fail, return NaT

def process_time_column(df, actual_time_col, prefix):
    # List of possible datetime formats
    time_formats = ['%M:%S', '%H:%M:%S']
    date_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M %p', '%B %d, %Y %H:%M:%S']

    # Apply the parsing function to the column
    datetime = df[actual_time_col].apply(lambda x: parse_datetime_column(x, date_formats, time_formats))

    if pd.notnull(datetime.iloc[0]) and isinstance(datetime.iloc[0], pd.Timestamp):
        if pd.notnull(datetime.dt.date.iloc[0]):
            if 'date' not in df.columns:
                df['date'] = datetime.dt.date
            datetime = datetime.dt.time

        df[prefix + 'hours'] = datetime.apply(lambda x: x.hour if pd.notnull(x) else np.nan)
        df[prefix + 'minutes'] = datetime.apply(lambda x: x.minute if pd.notnull(x) else np.nan)
        df[prefix + 'seconds'] = datetime.apply(lambda x: x.second if pd.notnull(x) else np.nan)
        
        df[prefix + 'total_seconds'] = df[prefix + 'hours'] * 3600 + df[prefix + 'minutes'] * 60 + df[prefix + 'seconds']
        df[prefix + 'time_scaled'] = 2 * (df[prefix + 'total_seconds'] / 86400) - 1
        
        df.drop([actual_time_col, prefix + 'hours', prefix + 'minutes', prefix + 'seconds', prefix + 'total_seconds'], axis=1, inplace=True)

        return ['time_scaled']
    return []

def process_date_column(df, actual_date_col, prefix):
    datetime = pd.to_datetime(df[actual_date_col], errors='coerce')
    
    if pd.notnull(datetime.iloc[0]):
        df['day_of_year'] = datetime.dt.dayofyear
        
        df[prefix + 'day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df[prefix + 'day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        df.drop([actual_date_col, 'day_of_year'], axis=1, inplace=True)
        return ['day_of_year_sin', 'day_of_year_cos']
    return []

def process_month_column(df, actual_month_col, prefix):
    df[actual_month_col] = pd.to_numeric(df[actual_month_col], errors='coerce')
    
    if df[actual_month_col].min() >= 1 and df[actual_month_col].max() <= 12:
        df[prefix + 'month_sin'] = np.sin(2 * np.pi * df[actual_month_col] / 12)
        df[prefix + 'month_cos'] = np.cos(2 * np.pi * df[actual_month_col] / 12)
        
        df.drop([actual_month_col], axis=1, inplace=True)
        return ['month_sin', 'month_cos']
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
    # Process columns
    if actual_time_col:
        processed_columns += process_time_column(df, actual_time_col, prefix)

    actual_date_col = next((col for col in date_cols if col in df.columns), None)
    if actual_date_col:
        processed_columns += process_date_column(df, actual_date_col, prefix)

    actual_month_col = next((col for col in month_cols if col in df.columns), None)
    if actual_month_col:
        processed_columns += process_month_column(df, actual_month_col, prefix)

    encoded_columns = convert_to_indices(df, processed_columns)
    return df, encoded_columns        
