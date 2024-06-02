'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple

def to_indices(df: pd.DataFrame, *columns: Optional[Union[List[str], pd.Index, str]]) -> Tuple[pd.Index, ...]:
    """
    Convert input columns to pd.Index format, handling lists of strings, pd.Index, None, or pd.Series.
    """
    def convert_to_index(col):
        if col is None:
            return pd.Index([])
        elif isinstance(col, list):
            return pd.Index(col)
        elif isinstance(col, pd.Index):
            return col
        elif isinstance(col, str):
            return pd.Index([col])
        elif isinstance(col, pd.Series) and col.name in df.columns:
            return pd.Index([col.name])
        else:
            raise ValueError("Input should be a list of strings, a pd.Index, a string, or a pandas Series")
    
    if len(columns) == 1:
        return convert_to_index(columns[0])
    return tuple(convert_to_index(col) for col in columns)

def remove_columns(df: pd.DataFrame, 
                   drop_columns: pd.Index,
                   exclude_columns: pd.Index = None) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame and drops rows with NaN values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    drop_columns (pd.Index): The columns to drop from the DataFrame.
    exclude_columns (pd.Index, optional): Columns to exclude from dropping.

    Returns:
    pd.DataFrame: The cleaned DataFrame with specified columns removed and NaN values dropped.
    """
    if exclude_columns is not None:
        drop_columns = drop_columns.difference(exclude_columns)
    
    if not drop_columns.empty:
        df = df.drop(columns=drop_columns)

    df_clean = df.dropna(axis=0).reset_index(drop=True).copy()
    
    return df_clean

def calculate_num_classes(target_df: pd.DataFrame) -> int:
    num_classes = 0
    target_col_len = len(target_df.columns)
    target_str_columns = target_df.select_dtypes(include=['object']).columns
    target_str_col_len = len(target_str_columns)
    non_str_target_len = target_col_len - target_str_col_len
    
    num_classes += non_str_target_len
    
    for col in target_str_columns:
        unique_values = target_df[col].nunique()
        num_classes += unique_values
    return num_classes

def display_statistics(df: pd.DataFrame) -> None:
    """
    Displays basic statistics for the input DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze.
    """
    
    # Calculate statistics
    min_values = df.min()
    max_values = df.max()
    mean_values = df.mean()
    std_values = df.std()
    null_counts = df.isnull().sum()

    # Combine all statistics into a single DataFrame
    stats_df = pd.DataFrame({
        'Min': min_values,
        'Max': max_values,
        'Mean': mean_values,
        'Std': std_values,
        'Null Count': null_counts
    })

    # Display the result in a Jupyter Notebook
    display(stats_df)  

def preprocess_cyclical_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Preprocesses specified cyclical columns in the DataFrame by applying sine and cosine transformations.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (List[str]): List of column names to be processed.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns processed.
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        
        # Apply sine and cosine transformations
        df[col + '_sin'] = np.sin(2 * np.pi * (df[col] - min_val) / range_val)
        df[col + '_cos'] = np.cos(2 * np.pi * (df[col] - min_val) / range_val)
        df.drop(col, axis=1, inplace=True)
    
    return df

def preprocess_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the 'date' column in the DataFrame by converting it to the day of the year
    and applying sine and cosine transformations to capture its cyclical nature.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
    pd.DataFrame: The DataFrame with the 'date' column processed.
    """
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Convert date to day of the year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Apply sine and cosine transformations
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df.drop(['date', 'day_of_year'], axis=1, inplace=True)
    
    return df

def preprocess_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the 'date' and 'time' columns in the DataFrame.
    - Converts the 'date' column to sine and cosine transformations of the month.
    - Encodes the 'time' column linearly from -1 to 1.
    - Handles cases where the 'time' column contains date information.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'date' and/or 'time' columns.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'date' and 'time' columns processed.
    """

    # Check if 'time' column also contains date information
    if 'time' in df.columns:
        # Try to parse the 'time' column with both date and time
        try:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['contains_date'] = df['time'].dt.date.notnull()  # Check if 'time' column has date info
            
            if df['contains_date'].all():  # If all entries in 'time' column have date info
                df['date'] = df['time'].dt.date  # Extract date part
                df['time'] = df['time'].dt.time  # Extract time part
        except:
            pass
        
        # Drop the helper column
        df.drop(columns=['contains_date'], inplace=True)
        
        # Handle 'time' column
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
        
        # Extract hours, minutes, and seconds
        df['hours'] = df['time'].dt.hour
        df['minutes'] = df['time'].dt.minute
        df['seconds'] = df['time'].dt.second
        
        # Calculate total seconds in the day
        df['total_seconds'] = df['hours'] * 3600 + df['minutes'] * 60 + df['seconds']
        
        # Encode time within the day linearly from -1 to 1
        df['time_scaled'] = 2 * (df['total_seconds'] / 86400) - 1
        
        # Drop the original 'time' column and the extracted 'hours', 'minutes', 'seconds', and 'total_seconds'
        df.drop(['time', 'hours', 'minutes', 'seconds', 'total_seconds'], axis=1, inplace=True)
        
    # Handle 'date' column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Extract month if 'month' column does not exist
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        
        # Drop the original 'date' column
        df.drop(['date'], axis=1, inplace=True)

    # Handle 'month' column (either pre-existing or extracted from 'date')
    if 'month' in df.columns:
        # Encode month using sine and cosine transformations
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop the extracted 'month' column
        df.drop(['month'], axis=1, inplace=True)
        
    return df