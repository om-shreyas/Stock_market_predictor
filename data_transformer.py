# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 09:30:17 2022

@author: Lenovo
"""

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

def data_transform(stock):
    company = yf.Ticker(stock)
    
    raw_data = company.history(period="max")
    
    temp_data = (raw_data.reset_index()).drop("Date",axis=1)
    temp_data = temp_data[:-1]
    
    temp_data["Answer Open"]=list(raw_data["Open"])[1:]
    temp_data["Answer Close"]=list(raw_data["Close"])[1:]
    
    return(temp_data)