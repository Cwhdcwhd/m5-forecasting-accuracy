import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

#step 1: load data
calendar = pd.read_csv("calendar.csv")
prices = pd.read_csv("sell_prices.csv")
sales = pd.read_csv("sales_train_validation.csv")

#step 2: convert categorical data to numerical data
calendar['event_name_1'] = pd.factorize(calendar['event_name_1'])[0]
calendar['event_type_1'] = pd.factorize(calendar['event_type_1'])[0]
calendar['event_name_2'] = pd.factorize(calendar['event_name_2'])[0]
calendar['event_type_2'] = pd.factorize(calendar['event_type_2'])[0]

prices['store_id'] = pd.factorize(prices['store_id'])[0]    
prices['item_id'] = pd.factorize(prices['item_id'])[0]

sales['item_id'] = pd.factorize(sales['item_id'])[0]
sales['dept_id'] = pd.factorize(sales['dept_id'])[0]
sales['cat_id'] = pd.factorize(sales['cat_id'])[0]
sales['store_id'] = pd.factorize(sales['store_id'])[0]
sales['state_id'] = pd.factorize(sales['state_id'])[0]

#step 3:create label and feature data
print("testing")
feature = pd.melt(sales,
             id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
             var_name='d',
             value_name='sales')
 
