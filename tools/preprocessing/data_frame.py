'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
    if not drop_columns.empty:
        if exclude_columns is not None:
            drop_columns = drop_columns.difference(exclude_columns)
        df = df.drop(columns=drop_columns)

    if not df.dropna().empty:
        print("before", df.isnull().sum())
    # Check for missing values and drop them
    df[:] = df[:].dropna(axis=0).reset_index(drop=True)
    if not df.dropna().empty:
        print("after", df.isnull().sum())

    return df

def encode_categorical_columns(df: pd.DataFrame, exclude_columns: pd.Index = None) -> Tuple[pd.DataFrame, dict]:
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

def scale_columns(df: pd.DataFrame, 
                  minmax_columns: pd.Index, 
                  standard_columns: pd.Index, 
                  robust_columns: pd.Index,
                  exclude_columns: pd.Index) -> pd.DataFrame:

    scaler_dict = {
        'MinMax': MinMaxScaler,
        'Standard': StandardScaler,
        'Robust': RobustScaler
    }
        
    for columns, scaler in zip([minmax_columns, standard_columns, robust_columns], 
                               ['MinMax', 'Standard', 'Robust']):
        valid_columns = columns.difference(exclude_columns)
        valid_columns = valid_columns.intersection(df.columns)
        if not valid_columns.empty:
            # Remove missing columns
            scaler_instance = scaler_dict[scaler]()
            df[valid_columns] = scaler_instance.fit_transform(df[valid_columns])
    
    return df

def process_df(df: pd.DataFrame, 
                 drop_columns: pd.Index,
                 one_hot_columns: pd.Index,
                 minmax_columns: pd.Index,
                 standard_columns: pd.Index, 
                 robust_columns: pd.Index) -> pd.DataFrame:
    
    # First, drop unwanted columns using the new function
    df = remove_columns(df, drop_columns)

    if not one_hot_columns.empty:
        df = pd.get_dummies(df, columns=one_hot_columns, drop_first=False).astype(float)

    df, encoded_columns = encode_categorical_columns(df)
    
    non_scale_columns = one_hot_columns.union(encoded_columns)
    df = scale_columns(df, minmax_columns, standard_columns, robust_columns, exclude_columns=non_scale_columns)
    
    return df

def process_dataframe(df: pd.DataFrame, target_columns, **kwargs) -> pd.DataFrame:
    
    drop_columns = kwargs.get('drop_columns', pd.Index([]))
    one_hot_columns = kwargs.get('one_hot_columns', pd.Index([]))
    minmax_columns = kwargs.get('minmax_columns', pd.Index([]))
    standard_columns = kwargs.get('standard_columns', pd.Index([]))
    robust_columns = kwargs.get('robust_columns', pd.Index([]))

    target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns = \
        to_indices(df, target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns)
    
    target_df = df[target_columns]
    df.drop(columns=target_columns, inplace=True)
    
    ################## Data Preprocessing #####################
    df = process_df(df, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns)
    df[:] = df[:].astype(float)

    ################## Target Preprocessing ###################
    target_df = process_df(target_df, pd.Index([]), pd.Index([]), minmax_columns, standard_columns, robust_columns)

    # Calculate the number of features and classes
    num_features = df.shape[1]
    num_classes = target_df.shape[1]

    # Concatenate target columns to the end
    df = pd.concat([df, target_df], axis=1)   

    ##################### Description ##########################
    

    description = {}
    description['num_features'] = num_features
    description['num_classes'] = num_classes
    
    return df, description