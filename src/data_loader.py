# -*- coding: utf-8 -*-
"""
Filename: data_loader.py
Author: Andrew Francey
Date: 2026-01-29
Description: File containing functions to load and handle the datasets.
Version: 1.0.0
License: Proprietary Licesneses
Dependencies: os, pandas
Usage: Load functions from this module import as 
    from data_loader import ...
"""

import os
import pandas as pd

def load_creditcard_data(train_path="../Data/fraudTrain.csv", test_path="../Data/fraudTest.csv"):
    """
    Load the Kaggle Credit Card Fraud Detection pre-split dataset and combine them
    into a single dataframe.

    Parameters
    ----------
    train_path : str
        File path to the creditcard.csv training dataset. Default is 'Data/fraudTrain.csv'
    test_path : str
        File path to the creditcard.csv testing dataset. Default is 'Data/fraudTest.csv'

    Returns
    -------
    pd.DataFrame
        A single pandas dataframe for all the creditcard dataset.  
    """
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
    
    # Load in the data for the train and test datasets.
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine both the train and test datasets.
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    return df

