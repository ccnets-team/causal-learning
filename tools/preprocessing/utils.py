'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import re
import pandas as pd
import numpy as np
from IPython.display import display
from typing import Optional, List, Union, Tuple
PROCESSED_PREFIX = "ccnets_processed_"

def convert_to_indices(df: pd.DataFrame, *columns: Optional[Union[List[str], pd.Index, str]]) -> Tuple[pd.Index, ...]:
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

def fill_random(df: pd.DataFrame, column: str):
    """Fill NaN values in the specified column with random choices from its non-NaN values."""
    non_missing = df[column].dropna().unique()
    df[column] = df[column].apply(lambda x: np.random.choice(non_missing) if pd.isna(x) else x)

def handle_missing_values(df: pd.DataFrame, 
                   drop_columns: pd.Index,
                   exclude_columns: pd.Index = None,
                   fill_random_columns: list = None) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame, and fills specified columns with random non-NaN values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    drop_columns (pd.Index): The columns to drop from the DataFrame.
    exclude_columns (pd.Index, optional): Columns to exclude from dropping.
    fill_random_columns (list, optional): List of columns to fill NaN values with random non-NaN data.

    Returns:
    pd.DataFrame: The modified DataFrame with specified columns removed and specified NaNs filled randomly.
    """
    if exclude_columns is not None:
        drop_columns = drop_columns.difference(exclude_columns)
    
    if not drop_columns.empty:
        print(f"Dropped columns: {', '.join(drop_columns)}")
        df = df.drop(columns=drop_columns)

    if fill_random_columns:
        for column in fill_random_columns:
            if column in df.columns:
                fill_random(df, column)

    return df

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

def display_statistics(df: pd.DataFrame, description: dict) -> None:
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
    
    stats_df['Scaled'] = stats_df.index.map(lambda x: description['scalers'].get(x, None))
    stats_df['Scaled'] = stats_df['Scaled'].apply(lambda x: x.capitalize() if x else None)
    
    one_hot_mask = stats_df.index.isin(description['one_hot_encoded_columns'])
    datetime_encode_mask = stats_df.index.isin(description['encoded_datatime_columns'])
    
    stats_df['Encoded'] = None

    update_one_hot_encoded_columns(stats_df, description['one_hot_encoded_columns'])
    stats_df.loc[one_hot_mask, 'Encoded'] = 'One-hot'
    stats_df.loc[datetime_encode_mask, 'Encoded'] = 'EncodedDateTime'
    
    # Display the result in a Jupyter Notebook
    display(stats_df)  

def get_columns(**kwargs) -> Tuple[pd.Index, pd.Index, pd.Index, pd.Index, pd.Index, pd.Index, pd.Index]:
    """
    Get the columns from the DataFrame based on the target columns and keyword arguments.
    """
    drop_columns = kwargs.get('drop_columns', pd.Index([]))
    one_hot_columns = kwargs.get('one_hot_columns', pd.Index([]))
    
    return drop_columns, one_hot_columns

def generate_description(**kwargs) -> dict:
    """
    Generates a description dictionary from the provided keyword arguments.
    
    Parameters:
    **kwargs: Arbitrary keyword arguments to be included in the description.
    
    Returns:
    dict: A dictionary containing the description information.
    """
    description = {}
    for key, value in kwargs.items():
        description[key] = value
    return description

def remove_process_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the 'ccnets_processed_' prefix from DataFrame columns and handles duplicates.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with updated column names.
    """
    # Before detaching the prefix, check for duplicate columns
    columns_with_prefix = df.columns
    columns_without_prefix = [col.split(PROCESSED_PREFIX)[-1] for col in columns_with_prefix]

    # Check for duplicates
    duplicates = set([col for col in columns_without_prefix if columns_without_prefix.count(col) > 1])

    # Rename duplicates with a suffix
    if duplicates:
        for dup in duplicates:
            count = 1
            for i in range(len(columns_without_prefix)):
                if columns_without_prefix[i] == dup:
                    columns_without_prefix[i] = f"{dup}_dup{count}"
                    count += 1
                    
    if len(duplicates) > 0:
        print("Duplicate columns were detected and renamed with a suffix.")
    
    # Assign the new column names to the DataFrame
    df.columns = columns_without_prefix
    
    return df
def update_one_hot_encoded_columns(stats_df, one_hot_encoded_columns):
    for base_col in one_hot_encoded_columns:
        pattern = re.compile(rf'^{base_col}_\d+$')  
        for col in stats_df.index:
            if pattern.match(col):
                stats_df.loc[col, 'Encoded'] = 'One_hot'
                
