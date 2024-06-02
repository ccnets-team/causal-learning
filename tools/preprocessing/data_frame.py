'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

import pandas as pd
from typing import Tuple
from tools.preprocessing.scaler import scale_columns
from tools.preprocessing.encode import encode_data_columns, encode_label_columns
from tools.preprocessing.utils import calculate_num_classes, to_indices, remove_columns, display_statistics
from tools.preprocessing.utils import preprocess_cyclical_columns, preprocess_datetime_columns

def process_df(df: pd.DataFrame, 
                 one_hot_columns: pd.Index,
                 minmax_columns: pd.Index,
                 standard_columns: pd.Index, 
                 robust_columns: pd.Index, 
                 exclude_scale_columns: pd.Index) -> Tuple[pd.DataFrame, dict]:
    
    original_columns = df.columns
    
    if not one_hot_columns.empty:
        df = pd.get_dummies(df, columns=one_hot_columns, drop_first=False).astype(float)

    df, encoded_columns = encode_data_columns(df)
        
    non_scale_columns = one_hot_columns.union(encoded_columns).union(exclude_scale_columns)
    df, scaler_description = scale_columns(df, original_columns, minmax_columns, standard_columns, robust_columns, exclude_columns=non_scale_columns)
    
    return df, encoded_columns, scaler_description
    
def process_dataframe(df: pd.DataFrame, target_columns, **kwargs) -> Tuple[pd.DataFrame, dict]:
    
    drop_columns = kwargs.get('drop_columns', pd.Index([]))
    one_hot_columns = kwargs.get('one_hot_columns', pd.Index([]))
    minmax_columns = kwargs.get('minmax_columns', pd.Index([]))
    standard_columns = kwargs.get('standard_columns', pd.Index([]))
    robust_columns = kwargs.get('robust_columns', pd.Index([]))
    exclude_scale_columns = kwargs.get('exclude_scale_columns', pd.Index([]))

    target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns, exclude_scale_columns = \
        to_indices(df, target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns, exclude_scale_columns)

    exclude_scale_columns = exclude_scale_columns.union(df.columns[df.columns.str.startswith('ccnets_processed')])

    # First, drop unwanted columns using the new function
    df = remove_columns(df, drop_columns)
        
    target_df = df[target_columns]
    df.drop(columns=target_columns, inplace=True)
    
    # Process the DataFrame excluding the target columns
    df, encoded_columns, scaler_description = process_df(df, one_hot_columns, minmax_columns, standard_columns, robust_columns, exclude_scale_columns)

    # Convert the entire DataFrame to float
    df = df.astype(float)
    num_features = df.shape[1]
    num_classes = calculate_num_classes(target_df)
    
    # Process the target columns separately
    target_df, target_encoded_columns = encode_label_columns(target_df)

    # Concatenate target columns to the end
    df = pd.concat([df, target_df], axis=1)   

    ##################### Description ##########################
    description = {}
    description['num_features'] = num_features
    description['num_classes'] = num_classes
    description['encoded_columns'] = encoded_columns
    description['target_encoded_columns'] = target_encoded_columns
    description['scalers'] = scaler_description
    
    display_statistics(df)

    return df, description