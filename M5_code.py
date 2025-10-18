import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import gc

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

print("Loading data...")
calendar = pd.read_csv('calendar.csv')
prices = pd.read_csv('sell_prices.csv')
sales = pd.read_csv('sales_train_evaluation.csv')
calendar.drop(columns=['date'], inplace=True)
print("Creating features and labels...")
columns = [f'd_{i}' for i in range(1942, 1970)]

# Number of rows you want
num_rows = 39490
df = pd.DataFrame(0, index=range(num_rows), columns=columns)
sales= pd.concat([sales, df], axis=1)
calendar = downcast(calendar)
prices = downcast(prices)
sales = downcast(sales)



#Create the DataFrame filled with zeros


feature = pd.melt(sales,
                  id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                  var_name='d',
                  value_name='sales')
feature = pd.merge(feature, calendar, on='d', how='left')
feature = pd.merge(feature, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
del calendar, prices, sales

# Factorize categorical columns after filling missing
for c in ['store_id','item_id','dept_id','cat_id','state_id',
          'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','weekday']:
    feature[c] = pd.factorize(feature[c])[0]
feature['d'] = feature['d'].str.split('_').str[1].astype(int)
print("Preparing data for training/testing...")

train_mask = feature['d'] <= 1913  # train days <= 1913
test_mask = (feature['d'] >= 1914) & (feature['d'] <= 1941)  # test days 1914 to 1941
predict_mask = feature['d'] > 1941  # prediction days > 1941

print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}, Predict samples: {predict_mask.sum()}")

y = feature['sales']
X = feature.drop(columns=['id', 'sales','item_id'])
del feature

# Fill missing values to avoid errors
y = y.fillna(0)
X = X.fillna(0)

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]
predict_idx = np.where(predict_mask)[0]


# Optimized XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',        # faster histogram-based algorithm
    'nthread': 8,                 # set according to your CPU cores
    'max_depth': 6,               # reduce depth for speed
    'subsample': 0.8,             # row sampling
    'colsample_bytree': 0.8       # feature sampling
}

chunk_size = 20000  # larger chunks to reduce overhead
booster = None

print("Training XGBoost model in chunks with progress monitoring...")
for start in range(0, len(train_idx), chunk_size):
    end = min(start + chunk_size, len(train_idx))
    train_chunk_idx = train_idx[start:end]
    dtrain_chunk = xgb.DMatrix(X.iloc[train_chunk_idx].values, label=y.iloc[train_chunk_idx].values)
    
    booster = xgb.train(
        params,
        dtrain_chunk,
        num_boost_round=10,
        xgb_model=booster,
        evals=[(dtrain_chunk, 'train_chunk')],
        verbose_eval=1
    )
    
    del dtrain_chunk
    gc.collect()

print("Model trained.")

dtest = xgb.DMatrix(X.iloc[test_idx].values, label=y.iloc[test_idx].values)
dpredict = xgb.DMatrix(X.iloc[predict_idx].values)

print("Making predictions on test set...")
preds_test = booster.predict(dtest)
rmse = np.sqrt(np.mean((preds_test - y.iloc[test_idx]) ** 2))
print(f"Test RMSE: {rmse}")

print("Plotting feature importance...")
xgb.plot_importance(booster)
plt.show()

print("Making predictions on prediction set...")
preds = booster.predict(dpredict)
print(f"Prediction set predictions: {preds}", preds.shape)

print("Saving model...")
booster.save_model('xgb_model.json')
print("Model saved as xgb_model.json")
