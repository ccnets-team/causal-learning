'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from preprocessing.utils import convert_to_indices
from preprocessing.utils import PROCESSED_PREFIX 

def encode_label_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encodes label columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    Tuple[pd.DataFrame, dict]: The DataFrame with encoded label columns and a dictionary of encoded columns.
    """
    
    # Identify string-type columns
    str_columns = df.select_dtypes(include=['object']).columns
    
    # Automatically exclude columns that already have the prefix
    exclude_columns = df.columns[df.columns.str.startswith(PROCESSED_PREFIX)]
    str_columns = str_columns.difference(exclude_columns)

    # Process string-type columns
    for col in str_columns:
        unique_values = df[col].nunique()
        print(f"Column '{col}' has {unique_values} unique values.")
        # Convert string columns to numeric values from 0 to n
        le = LabelEncoder()
        df[PROCESSED_PREFIX + col] = le.fit_transform(df[col])
        df.drop(columns=[col], inplace=True)
    
    encoded_columns = convert_to_indices(df, [PROCESSED_PREFIX + col for col in str_columns.tolist()])
    
    return df, encoded_columns

def detect_categorical_named_columns(df):
    """
    Detects columns likely to be categorical based on naming conventions.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    list: List of column names identified as categorical.
    """
    categorical_columns = []
    suffixes = ['_id', '_code', '_type', '_cat', '_category', '_status', '_flag', \
                'Id', 'Code', 'Type', 'Category', 'Status', 'Flag']
    prefixes = ['id_', 'code_', 'type_', 'cat_', 'category_', 'status_', 'flag_', 'is_', 'has_', \
                'Id', 'Code', 'Type', 'Category', 'Status', 'Flag', 'Is', 'Has', 'is', 'has']

    suffix_set = set(suffixes)
    prefix_set = set(prefixes)
    
    for column in df.columns:
        if any(column.endswith(suffix) for suffix in suffix_set) or any(column.startswith(prefix) for prefix in prefix_set):
            categorical_columns.append(column)

    return categorical_columns


def one_hot_encode_columns(df: pd.DataFrame, encode_columns: list) -> Tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    given_one_hot_columns (list): Columns to one-hot encode.
    
    Returns:
    Tuple[pd.DataFrame, dict]: The processed DataFrame and a dictionary of encoded columns.
    """

    # Identify string-type columns
    str_columns = df.select_dtypes(include=['object']).columns

    name_indicated_columns = detect_categorical_named_columns(df)
    
    category_columns = str_columns.union(encode_columns).union(name_indicated_columns)

    # Automatically exclude columns that already have the prefix
    exclude_columns = df.columns[df.columns.str.startswith(PROCESSED_PREFIX)]
    
    category_columns = category_columns.difference(exclude_columns)
    
    # Lists to hold names of columns that will be converted
    binary_list = []
    one_hot_list = []

    # Process string-type columns based on the number of unique values
    for col in category_columns:
        unique_values = df[col].nunique()
        print(f"Column '{col}' has {unique_values} unique values.")
        if unique_values == 1:
            # Drop columns with only 1 unique value
            df = df.drop(columns=[col])
        elif unique_values == 2:
            # Mark columns with exactly 2 unique values for binary conversion
            binary_list.append(col)
        else:
            # Mark columns with more than 2 unique values for one-hot encoding
            one_hot_list.append(col)
    
    # Binary encoding for columns with exactly 2 unique values
    for col in binary_list:
        df[PROCESSED_PREFIX + col] = pd.get_dummies(df[col], drop_first=True).astype(float)
        df = df.drop(columns=[col])

    # One-hot encoding for columns with more than 2 unique values
    if one_hot_list:
        df = pd.get_dummies(df, columns=one_hot_list, prefix=[PROCESSED_PREFIX + col for col in one_hot_list], drop_first=False).astype(float)
        
    encoded_columns = convert_to_indices(df, [col for col in binary_list] + [col for col in one_hot_list])
    return df, encoded_columns
