# m5-forecasting-accuracy
Overview
This code implements a forecasting pipeline for the M5 retail sales dataset using XGBoost, a gradient boosting decision tree algorithm. It processes raw sales, calendar, and price data, engineer features, and trains an incremental XGBoost model on large-scale data using chunked training to manage memory efficiently.
________________________________________
#Data Loading and Memory Optimization
#•	Files loaded: calendar.csv (calendar features), sell_prices.csv (prices data), and sales_train_evaluation.csv (historical sales).
#•	The downcast() function converts DataFrame column data types to more memory-efficient ones (e.g., int64 to int32, float64 to float32, objects to category) to reduce RAM usage.

Feature Engineering
•	Additional zero sales columns (d_1942 to d_1969) are appended to extend historical sales to future days.
•	Dataset is transformed from wide format to long format via pd.melt() to consolidate daily sales into rows with identifier columns (id, item_id, etc.).
•	Calendar and prices data merged into features by keys (d, store_id, item_id, wm_yr_wk).
•	Categorical variables are factorized after filling missing data, converting text categories into numerical codes.
•	The day column 'd' is extracted as an integer for better indexing.

Dataset Splitting
•	The dataset is split into:
•	Training: days ≤ 1913
•	Testing: days between 1914 and 1941
•	Prediction set: days > 1941

Masks are created accordingly and converted into index arrays.
Data Preparation
•	Target variable y is sales, with missing values filled as zeros.
•	Features X exclude identifiers and sales columns.
•	Both X and y are cast to 32-bit floats to save memory.

Model Training
•	XGBoost Parameters:
•	Objective: squared error regression
•	Evaluation metric: RMSE
•	Tree method: hist (histogram based for faster training)
•	Multithreading enabled (nthread=8, adjust to your CPU)
•	Subsampling of rows and columns for speed and regularization
•	Maximum tree depth set to 6 to balance accuracy and speed
•	Chunked Training:
•	The training data is split into chunks (size 20,000 rows) to avoid memory overload.
•	For each chunk, a DMatrix is created and the model incrementally trained using previous booster state.
•	Training progress is printed with RMSE after each boosting round.
•	Garbage collection runs after processing each chunk to free up memory.

Model Evaluation and Prediction
•	After training, the model predicts sales on the test set.
•	RMSE computed on test predictions to measure accuracy.

Visualization and Output
•	Feature importance plot displays the relative contribution of each feature to the model.
•	Predictions on the future days (prediction set) are generated and shape printed.
•	The trained model is saved to xgb_model.json for later use.

Key Benefits of This Approach
•	Efficient memory usage via downcasting and chunked training enables handling large-scale retail sales data.
•	Incremental training supports datasets too large to fit in memory at once.
•	Use of XGBoost’s histogram method accelerates training compared to exact greedy methods.
•	Model explainability via feature importance helps understand key drivers (such as price, events, and calendar features).






