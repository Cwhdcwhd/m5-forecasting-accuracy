import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif 'object' in str(t):
            if cols[i] != 'date':
                df[cols[i]] = df[cols[i]].astype('category')
    return df
#step 1: load data
print("loading data")
# calendar_url = 'https://media.githubusercontent.com/media/Cwhdcwhd/m5-forecasting-accuracy/refs/heads/main/calendar.csv'
# prices_url = 'https://media.githubusercontent.com/media/Cwhdcwhd/m5-forecasting-accuracy/refs/heads/main/sell_prices.csv'
# sales_url = 'https://media.githubusercontent.com/media/Cwhdcwhd/m5-forecasting-accuracy/refs/heads/main/sales_train_validation.csv'

calendar = pd.read_csv('calendar.csv')
print("calendar loaded")
prices = pd.read_csv('sell_prices.csv')
print("prices loaded")
sales = pd.read_csv('sales_train_validation.csv')
print("sales loaded")
calendar.drop(columns=['date'], inplace=True)
calendar = downcast(calendar)
prices = downcast(prices)
sales = downcast(sales)

#step 2:create label and feature data
print("start creating label and feature data")
feature = pd.melt(sales,
             id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
             var_name='d',
             value_name='sales')
feature = pd.merge(feature, calendar, on='d', how='left')
feature = pd.merge(feature, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
del calendar, prices, sales
print("label and feature data created")

#step 3: convert categorical data to numerical data
print("start factorizing")
for c in ['store_id','item_id','dept_id','cat_id','state_id','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','weekday']:
    feature[c] = pd.factorize(feature[c])[0]
feature['d'] = feature['d'].str.split('_').str[1].astype(int)
print (feature.shape)
print("factorizing done")

# #step 4: prepare data for xgboost
print("start preparing data for xgboost")
y = feature['sales']
X = feature.drop(columns=['id', 'sales'])


train_mask = feature['d'] < 1914
test_mask = feature['d'] >= 1914
del feature
print("preparing done")



#step 5: train xgboost model
print("start training xgboost model")   
dtrain = xgb.DMatrix(X[train_mask], label=y[train_mask])
dtest = xgb.DMatrix(X[test_mask], label=y[test_mask])
model=xgb.train({'objective':'reg:squarederror', 'eval_metric':'rmse'}, dtrain, num_boost_round=100)
print("xgboost model trained")

#step 6: make predictions
print("start making predictions")
preds = model.predict(dtest)
print("predictions made")

#step 7: evaluate model
rmse = np.sqrt(np.mean((preds - y[test_mask]) ** 2))
print(f"RMSE: {rmse}")

#step 8: plot feature importance
xgb.plot_importance(model)
plt.show()

#step 9: save model
model.save_model('xgb_model.json')
print("model saved as xgb_model.json")
