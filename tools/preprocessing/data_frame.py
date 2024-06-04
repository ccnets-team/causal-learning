'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import numpy as np
import pandas as pd
from typing import Tuple, List, Tuple
from tools.preprocessing.utils import generate_description, remove_process_prefix
from tools.preprocessing.utils import calculate_num_classes, convert_to_indices, remove_columns, display_statistics
from tools.preprocessing.scaler import auto_scale_columns
from tools.preprocessing.encode import one_hot_encode_columns, encode_label_columns
from tools.preprocessing.datetime import auto_encode_datetime_columns
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

def auto_preprocess_dataframe(df: pd.DataFrame, target_columns, drop_columns = None, encode_columns = None, no_scale_columns = None) -> Tuple[pd.DataFrame, dict]:
    """
    Automatically preprocesses the DataFrame by encoding, scaling, and handling target columns.
    """

    # Convert columns to DataFrame indices
    target_columns, drop_columns, encode_columns, no_scale_columns = \
        convert_to_indices(df, target_columns, drop_columns, encode_columns, no_scale_columns)

    # Drop unwanted columns
    df = remove_columns(df, drop_columns)
    
    # Split DataFrame into feature columns and target columns
    df_x, df_y = df.drop(columns=target_columns), df[target_columns]
    
    df_x, encoded_datatime_columns = auto_encode_datetime_columns(df_x)
    
    # Encode non-target columns
    df_x, one_hot_encoded_columns = one_hot_encode_columns(df_x, encode_columns)
    
    # Scale non-target columns
    df_x, scaler_description = auto_scale_columns(df_x, no_scale_columns)
    
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
    encoded_columns = encoded_datatime_columns.union(one_hot_encoded_columns).union(encoded_target_columns)
    
    # Generate description dictionary
    description = generate_description(num_features=num_features, num_classes=num_classes,
                                       encoded_columns=encoded_columns,
                                       one_hot_encoded_columns=one_hot_encoded_columns, 
                                       encoded_datatime_columns=encoded_datatime_columns,
                                       scalers=scaler_description)
    
    # Display DataFrame statistics
    display_statistics(df, description)

    # Return processed DataFrame and description
    return df, description
