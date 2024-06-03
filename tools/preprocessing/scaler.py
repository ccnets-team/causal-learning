'''
Author:
        
        PARK, JunHo, junho@ccnets.org

        
        KIM, JoengYoong, jeongyoong@ccnets.org
        
    COPYRIGHT (c) 2024. CCNets. All Rights reserved.
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import skew
from tools.preprocessing.utils import PROCESSED_PREFIX 

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
            
def auto_scale_columns(df: pd.DataFrame, 
                  minmax_columns: pd.Index, 
                  standard_columns: pd.Index, 
                  robust_columns: pd.Index) -> Tuple[pd.DataFrame, dict]:

    scaler_dict = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    exclude_columns = df.columns[df.columns.str.startswith(PROCESSED_PREFIX)]
    
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
    remaining_columns = df.columns.difference(exclude_columns.union(minmax_columns).union(standard_columns).union(robust_columns))

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