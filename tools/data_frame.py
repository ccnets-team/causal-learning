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
                   target_columns: pd.Index, 
                   drop_columns: pd.Index) -> pd.DataFrame:
    if not drop_columns.empty:
        valid_drop_columns = drop_columns.difference(target_columns)
        df = df.drop(columns=valid_drop_columns)

    # Check for missing values and drop them
    df = df.dropna(axis=0).reset_index(drop=True)
    return df

def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # Identify string-type columns
    str_columns = df.select_dtypes(include=['object']).columns

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
                  non_scale_columns: pd.Index, 
                  minmax_columns: pd.Index, 
                  standard_columns: pd.Index, 
                  robust_columns: pd.Index) -> pd.DataFrame:
    # Initialize scalers
    if not minmax_columns.empty:
        minmax_scaler = MinMaxScaler()
        valid_minmax_columns = minmax_columns.difference(non_scale_columns)
        df[valid_minmax_columns] = minmax_scaler.fit_transform(df[valid_minmax_columns])

    if not standard_columns.empty:
        standard_scaler = StandardScaler()
        valid_standard_columns = standard_columns.difference(non_scale_columns.union(minmax_columns))
        df[valid_standard_columns] = standard_scaler.fit_transform(df[valid_standard_columns])

    if not robust_columns.empty:
        robust_scaler = RobustScaler()
        valid_robust_columns = robust_columns.difference(non_scale_columns.union(minmax_columns).union(standard_columns))
        df[valid_robust_columns] = robust_scaler.fit_transform(df[valid_robust_columns])

    # Identify columns that haven't been converted and are not target columns
    all_used_columns = non_scale_columns.union(minmax_columns).union(standard_columns).union(robust_columns)
    unchanged_columns = df.columns.difference(all_used_columns)
   
    # Inline float conversion
    df[unchanged_columns] = df[unchanged_columns].astype(float)
    return df

def process_dataframe(df: pd.DataFrame, 
                      target_columns: Optional[Union[List[str], pd.Index, str]],
                      drop_columns: Optional[Union[List[str], pd.Index, str]] = None,
                      one_hot_columns: Optional[Union[List[str], pd.Index, str]] = None,
                      minmax_columns: Optional[Union[List[str], pd.Index, str]] = None,
                      standard_columns: Optional[Union[List[str], pd.Index, str]] = None, 
                      robust_columns: Optional[Union[List[str], pd.Index, str]] = None) -> pd.DataFrame:
    # Convert inputs to pd.Index
    target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns = \
        to_indices(df, target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns)
    
    # First, drop unwanted columns using the new function
    df = remove_columns(df, target_columns, drop_columns)
    
    # One-hot encoding for columns with more than 2 unique values
    df = pd.get_dummies(df, columns=one_hot_columns, drop_first=False).astype(float)

    # Encode categorical columns
    df, encoded_columns = encode_categorical_columns(df)
 
    # Create a set of columns
    non_scale_columns = one_hot_columns.union(encoded_columns)
    
    df = scale_columns(df, non_scale_columns, minmax_columns, standard_columns, robust_columns)
    
    # Move target columns to the end
    target_data = df[target_columns]
    df.drop(columns=target_columns, inplace=True)
    df = pd.concat([df, target_data], axis=1)   

    return df
