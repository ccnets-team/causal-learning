'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

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

def auto_determine_scaler(data, skew_threshold=0.5, mean_threshold=2.0, outlier_threshold=1.5):
    """
    Determines the appropriate scaler for data preprocessing based on the skewness, mean, and presence of outliers.
    
    Parameters:
    data (pd.Series or np.ndarray): The input data to be analyzed.
    skew_threshold (float): The threshold for skewness to determine the need for scaling.
    mean_threshold (float): The threshold for the mean to determine the scaling method.
    outlier_threshold (float): The multiplier used in the IQR method to detect outliers.
    
    Returns:
    str: 'minmax', 'standard', 'robust', or 'none' indicating the recommended scaler.
    """
    if isinstance(data, pd.Series):
        data = data.values
    elif not isinstance(data, (np.ndarray, list)):
        raise ValueError("Input should be a single column of data as pd.Series, np.ndarray, or list")
    
    # Calculate skewness
    data_skewness = skew(data)
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Detect outliers using the IQR method
    lower_bound = Q1 - (outlier_threshold * IQR)
    upper_bound = Q3 + (outlier_threshold * IQR)
    outliers = (data < lower_bound) | (data > upper_bound)
    has_outliers = np.any(outliers)
    
    column_mean = np.abs(data).mean() 
    
    # # Determine the scaler
    if abs(data_skewness) < skew_threshold:
        if column_mean > mean_threshold:
            return "minmax"
        else:
            return "none"
    else:
        if column_mean > mean_threshold:
            if has_outliers:
                return "robust"
            else:
                return "standard"
        else:
            return "none"
            
def scale_columns(df: pd.DataFrame, 
                  original_columns: pd.Index, 
                  minmax_columns: pd.Index, 
                  standard_columns: pd.Index, 
                  robust_columns: pd.Index,
                  exclude_columns: pd.Index) -> Tuple[pd.DataFrame, dict]:

    scaler_dict = {
        'minmax': lambda: MinMaxScaler(feature_range=(-1, 1)),
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    
    df_scaled = df.copy()
    scaler_description = {}

    for columns, scaler in zip([minmax_columns, standard_columns, robust_columns], 
                               ['minmax', 'standard', 'robust']):
        valid_columns = columns.difference(exclude_columns)
        if not valid_columns.empty:
            scaler_instance = scaler_dict[scaler]()
            df_scaled[valid_columns] = scaler_instance.fit_transform(df[valid_columns])
            for col in valid_columns:
                scaler_description[col] = scaler
    
    # Determine and apply scalers for columns not in any predefined list
    remaining_columns = original_columns.difference(minmax_columns.union(standard_columns).union(robust_columns).union(exclude_columns))

    for column in remaining_columns:
        scaler_type = auto_determine_scaler(df.loc[:, column])
        if scaler_type == 'none':
            # centering the data
            df_scaled.loc[:, [column]] -= df.loc[:, [column]].mean()
        elif scaler_type == 'minmax':
            column_mean = df[column].abs().mean() + 1e-8
            df_scaled[column] = 2.0 * (df_scaled[column] / column_mean) - 1.0            
        else:
            scaler_instance = scaler_dict[scaler_type]()
            df_scaled.loc[:, [column]] = scaler_instance.fit_transform(df.loc[:, [column]])
        scaler_description[column] = scaler_type
            
    return df_scaled, scaler_description

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
    
def process_dataframe(df: pd.DataFrame, target_columns, **kwargs) -> Tuple[pd.DataFrame, dict]:
    
    drop_columns = kwargs.get('drop_columns', pd.Index([]))
    one_hot_columns = kwargs.get('one_hot_columns', pd.Index([]))
    minmax_columns = kwargs.get('minmax_columns', pd.Index([]))
    standard_columns = kwargs.get('standard_columns', pd.Index([]))
    robust_columns = kwargs.get('robust_columns', pd.Index([]))
    exclude_scale_columns = kwargs.get('exclude_scale_columns', pd.Index([]))

    target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns, exclude_scale_columns = \
        to_indices(df, target_columns, drop_columns, one_hot_columns, minmax_columns, standard_columns, robust_columns, exclude_scale_columns)

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