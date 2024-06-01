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