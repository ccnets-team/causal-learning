'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from tools.preprocessing.utils import to_indices

def encode_label_columns(df: pd.DataFrame, exclude_columns: pd.Index = None) -> Tuple[pd.DataFrame, dict]:
    # Identify string-type columns
    str_columns = df.select_dtypes(include=['object']).columns
    if exclude_columns is not None:
        str_columns = str_columns.difference(exclude_columns)

    # Process string-type columns
    for col in str_columns:
        unique_values = df[col].nunique()
        print(f"Column '{col}' has {unique_values} unique values.")
        # Convert string columns to numeric values from 0 to n
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    encoded_columns = to_indices(df, str_columns.tolist())
    
    return df, encoded_columns

def encode_data_columns(df: pd.DataFrame, exclude_columns: pd.Index = None) -> Tuple[pd.DataFrame, dict]:
    # Identify string-type columns
    str_columns = df.select_dtypes(include=['object']).columns
    if exclude_columns is not None:
        str_columns = str_columns.difference(exclude_columns)

    # Lists to hold names of columns that will be converted
    binary_list = []
    one_hot_list = []

    # Process string-type columns based on the number of unique values
    for col in str_columns:
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
        df[col] = pd.get_dummies(df[col], drop_first=True).astype(float)

    # One-hot encoding for columns with more than 2 unique values
    if one_hot_list:
        df = pd.get_dummies(df, columns=one_hot_list, drop_first=False).astype(float)
        
    encoded_columns = to_indices(df, binary_list + one_hot_list)
    return df, encoded_columns