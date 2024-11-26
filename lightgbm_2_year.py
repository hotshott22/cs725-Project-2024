import pickle
import os
import gc

import pandas as pd  
import numpy as np  

import lightgbm as lgb  
import warnings 
from sklearn.metrics import mean_squared_error 

warnings.filterwarnings('ignore') 

import random  

# Define the path where the dataset will be saved (current folder)
dataset_path = 'sales_train_evaluation_long_2_year.pkl'
y_val_path='y_val_path_2_year.pkl'
x_val_path='x_val_path_2_year.pkl'
X_train_path='x_train_path_2_year.pkl'
y_train_path='y_train_path_2_year.pkl'
validation_set_path='validation_set_2year.pkl'
prediction_set_path='prediction_set_2yeaR.pkl'

# Function to generate the dataset
def generate_dataset(sales_train_evaluation_long):
   
    for lag in [7,28]:
        sales_train_evaluation_long[f'sales_lag_{lag}'] = sales_train_evaluation_long.groupby('id')['sales'].shift(lag)

    for window in [7,28]:
        sales_train_evaluation_long[f'rolling_sales_mean_{window}'] = sales_train_evaluation_long.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window).mean())
        sales_train_evaluation_long[f'rolling_sales_std_{window}'] = sales_train_evaluation_long.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window).std())

    
    print(f"len(sales_train_evaluation_long): {len(sales_train_evaluation_long)}")
    print(f"sales_train_evaluation_long dataset shape after lag and rolling window features merging: {sales_train_evaluation_long.shape}")
    print(sales_train_evaluation_long.head())

   
    train_set = sales_train_evaluation_long[sales_train_evaluation_long['d'].isin([f'd_{i}' for i in range(1942-730, 1914)])]

    # print("Train set sample:")
    # print(f"len(Train set): {len(train_set)}")
    train_set.head()
    print(f"len(train_set): {len(train_set)}")
    print(f"train_set dataset shape : {train_set.shape}")
    print(train_set.head())

    # Filter the validation set for d_1914 to d_1941
    validation_set = sales_train_evaluation_long[sales_train_evaluation_long['d'].isin([f'd_{i}' for i in range(1914, 1942)])]
    print("Validation set sample:")
    print(f"len(Validation set): {len(validation_set)}")
    validation_set.head()
    print(f"len(validation_set): {len(validation_set)}")
    print(f"validation_set dataset shape : {validation_set.shape}")
    print(validation_set.head())

    validation_set.info()


    
    forecast_days = [f'd_{i}' for i in range(1942, 1970)]
    forecast_df = pd.DataFrame({'d': forecast_days})

  
    prediction_set = sales_train_evaluation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    prediction_set = prediction_set.merge(forecast_df, how='cross')

  
    prediction_set = prediction_set.merge(calendar, on='d', how='left')

  
    prediction_set = prediction_set.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    print("Prediction set sample:")
    print(f"len(Prediction set): {len(prediction_set)}")
  
    print(f"prediction_set dataset shape : {prediction_set.shape}")
    print(prediction_set.head())


    prediction_set.info()

   
    train_set = reduce_mem_usage(train_set)
    validation_set = reduce_mem_usage(validation_set)
    prediction_set = reduce_mem_usage(prediction_set)


   
    feature_columns = [
        'sales_lag_7', 'sales_lag_28', 
        'rolling_sales_mean_7', 'rolling_sales_std_7', 
        'rolling_sales_mean_28', 'rolling_sales_std_28', 
        'sell_price', 'wday', 'month', 'year', 
        'snap_CA', 'snap_TX', 'snap_WI'
    ]

   
    categorical_features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 
        'state_id', 'weekday'
    ]

  
    X_train = train_set[feature_columns]
    y_train = train_set['sales']
    X_val = validation_set[feature_columns]
    y_val = validation_set['sales']

    print(f"len(X_train): {len(X_train)}")
    print(f"X_train dataset shape : {X_train.shape}")
    print(X_train.head())
    print(f"len(y_train): {len(y_train)}")
    print(f"y_train dataset shape : {y_train.shape}")
    print(y_train.head())

    print(f"len(X_val): {len(X_val)}")
    print(f"X_val dataset shape : {X_val.shape}")
    print(X_val.head())
    print(f"len(y_val): {len(y_val)}")
    print(f"y_val dataset shape : {y_val.shape}")
    print(y_val.head())
 
    return sales_train_evaluation_long,y_val,X_val,X_train,y_train,validation_set,prediction_set



def save_dataset(sales_train_evaluation_long,y_val,X_val,X_train,y_train,validation_set,prediction_set):
    sales_train_evaluation_long.to_pickle(dataset_path)
    y_val.to_pickle(y_val_path)
    X_val.to_pickle(x_val_path)
    X_train.to_pickle(X_train_path)
    y_train.to_pickle(y_train_path)
    validation_set.to_pickle(validation_set_path)
    prediction_set.to_pickle(prediction_set_path)
    
    print(f"Dataset saved to {dataset_path}")

# Function to load the dataset
def load_dataset():
    
    return pd.read_pickle(dataset_path),pd.read_pickle(y_val_path) ,pd.read_pickle(x_val_path),pd.read_pickle(X_train_path),pd.read_pickle(y_train_path),pd.read_pickle(validation_set_path),pd.read_pickle(prediction_set_path)

def seed_everything(seed):
    np.random.seed(seed)  
    random.seed(seed)  

seed_everything(seed=2024) 


calendar = pd.read_csv("calendar_mini_2_year.csv") 
print(f"len(calendar):{len(calendar)}") 
calendar.head()  
sales_train_evaluation = pd.read_csv("sales_train_evaluation_mini_2_year.csv")
print(f"len(sales_train_evaluation): {len(sales_train_evaluation)}")
sales_train_evaluation.head()
sell_prices = pd.read_csv("sell_prices_mini_2_year.csv")
print(f"len(sell_prices):{len(sell_prices)}")
sell_prices.head()

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2  # Initial memory usage in MB
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:  # Downcast numerics
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        elif col_type == 'object':  # Handle object types
            if col == 'date':  # Convert date column to datetime
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            else:
                df[col] = df[col].astype('category')  # Convert other object types to category
    end_mem = df.memory_usage().sum() / 1024**2  # Final memory usage in MB
    if verbose:
        print(f'Memory usage reduced to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

calendar = reduce_mem_usage(calendar)
sell_prices = reduce_mem_usage(sell_prices)
sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)

print(f"calendar dataset shape after optimization: {calendar.shape}")
print(f"sales_train_evaluation dataset shape after optimization: {sales_train_evaluation.shape}")
print(f"sell_prices dataset shape after optimization: {sell_prices.shape}")

if os.path.exists(dataset_path):
    use_existing = input(f"Do you want to use the existing dataset saved as '{dataset_path}'? (y/n): ").strip().lower()
else:
    use_existing = 'n'  # If the file doesn't exist, automatically choose to regenerate the dataset


if use_existing == 'y' and os.path.exists(dataset_path):
    print("Loading the existing dataset...")
    sales_train_evaluation_long,y_val,X_val,X_train,y_train,validation_set,prediction_set = load_dataset()
else:
   

    
    d_cols_eval = [f"d_{i}" for i in range(1942-730, 1942)] # for one years
    sales_train_evaluation_long = sales_train_evaluation.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=d_cols_eval,
        var_name="d",
        value_name="sales"
    )
    print(f"len(sales_train_evaluation_long): {len(sales_train_evaluation_long)}")
    print(f"sales_train_evaluation_long dataset shape after optimization: {sales_train_evaluation_long.shape}")
    print(sales_train_evaluation_long.head())
   


    sales_train_evaluation_long.head()
    sales_train_evaluation_long = sales_train_evaluation_long.merge(calendar, on="d", how="left")
    sales_train_evaluation_long.head()
    print(f"len(sales_train_evaluation_long): {len(sales_train_evaluation_long)}")
    print(f"sales_train_evaluation_long dataset shape after calender merging: {sales_train_evaluation_long.shape}")
    print(sales_train_evaluation_long.head())

    sales_train_evaluation_long = sales_train_evaluation_long.merge(
        sell_prices, 
        on=["store_id", "item_id", "wm_yr_wk"], 
        how="left"
    )
    print(f"len(sales_train_evaluation_long): {len(sales_train_evaluation_long)}")
    print(f"sales_train_evaluation_long dataset shape after sell_prices merging: {sales_train_evaluation_long.shape}")
    print(sales_train_evaluation_long.head())
    sales_train_evaluation_long.head()

   
    # Ask the user if they want to use the existing dataset or generate a new one

    print("Generating a new dataset...")
    sales_train_evaluation_long,y_val,X_val,X_train,y_train,validation_set,prediction_set = generate_dataset(sales_train_evaluation_long)
    save_dataset(sales_train_evaluation_long,y_val,X_val,X_train,y_train,validation_set,prediction_set)



feature_columns = [
        'sales_lag_7', 'sales_lag_28', 
        'rolling_sales_mean_7', 'rolling_sales_std_7', 
        'rolling_sales_mean_28', 'rolling_sales_std_28', 
        'sell_price', 'wday', 'month', 'year', 
        'snap_CA', 'snap_TX', 'snap_WI'
    ]

   
categorical_features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 
        'state_id', 'weekday'
    ]


categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]


train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature_indices)
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature_indices)



print(f"(train_data): ")
# print(f"train_data dataset shape : {train_data.shape}")
# print(train_data.head())

print(f"(model trining satartd): ")
# print(f"val_data dataset shape : {val_data.shape}")
# print(val_data.head())



lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.5,
    'seed' : 2000
}


# Define the path where the model will be saved (current folder)
model_path = 'lightgbm_model_2year_wo_tweeidie.txt'

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading existing model...")
    # Load the model if it already exists
    model = lgb.Booster(model_file=model_path)
else:
    print("Training new model...")
    # If the model does not exist, train a new model
    model = lgb.train(
        lgb_params, 
        train_data, 
        valid_sets=[train_data, val_data], 
        num_boost_round=1000,
        callbacks=[
                lgb.log_evaluation(100),  # Logs every 100 rounds
                lgb.early_stopping(stopping_rounds=50)  # Early stopping if no improvement
            ]
    )

    # Save the trained model to the current directory
    model.save_model(model_path)
    print(f"Model saved to {model_path}")






print(f"(model trining completed): ")

print(f"(validation started ): ")


# 7. Make non-recursive predictions for the entire validation period
val_predictions = model.predict(X_val)
validation_set['predicted_sales'] = val_predictions

print(f"(validation completed ): ")
print(f"len(validation_set): {len(validation_set)}")
print(f"validation_set dataset shape : {validation_set.shape}")
print(validation_set.head())

# Evaluate predictions
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"RMSE on validation set: {rmse}")

# Display the validation set with predictions
validation_set[['id', 'd', 'sales', 'predicted_sales']].head()


validation_set[15000:15010]

print(f"(validation data for sel ): ")
print(validation_set[8000:8010])

# Specify the item_id, dept_id, cat_id, store_id you want to filter
item_id = 'FOODS_2_075'
dept_id = 'FOODS_2'
cat_id = 'FOODS'
store_id = 'TX_1'


# Define the target range of `d` values for the validation period
valid_d_range = [f'd_{i}' for i in range(1914, 1942)]

# Filter the validation data for the specified item-store combination and `d` range
valid_data = validation_set[
    (validation_set['item_id'] == item_id) &
    (validation_set['dept_id'] == dept_id) &
    (validation_set['cat_id'] == cat_id) &
    (validation_set['store_id'] == store_id) &
    (validation_set['d'].isin(valid_d_range))
].copy()

print(f"(valid_data ): ")
print(f"len(valid_data): {len(valid_data)}")
print(f"valid_data dataset shape : {valid_data.shape}")
print(validation_set.head())



import plotly.graph_objects as go


fig = go.Figure()


fig.add_trace(go.Scatter(x=valid_data['d'], y=valid_data['sales'], mode='lines+markers', name='Actual Sales'))


fig.add_trace(go.Scatter(x=valid_data['d'], y=valid_data['predicted_sales'], mode='lines+markers', name='Predicted Sales'))


fig.update_layout(
    title=f"Actual vs Predicted Sales for (item_id={item_id}, dept_id={dept_id}, cat_id={cat_id}, store_id={store_id})",
    xaxis_title="Day",
    yaxis_title="Sales",
    legend_title="Legend",
    template="plotly_dark"
)


fig.write_image("sales_comparison_plot for validation set 2 year.png")








# Initialize recent sales pivot specifically for this item and store
recent_sales = validation_set[
    (validation_set['item_id'] == item_id) &
    (validation_set['dept_id'] == dept_id) &
    (validation_set['cat_id'] == cat_id) &
    (validation_set['store_id'] == store_id) &
    (validation_set['d'].isin([f'd_{i}' for i in range(1914, 1942)]))
][['id', 'd', 'sales']]




# Generate the forecast for the next 28 days
forecast_days = [f'd_{i}' for i in range(1942, 1970)]  # This range assumes you're forecasting for the next 28 days


print(f"(forecast_days 11 ): ")



# Filter prediction_set for only the selected item and store combination
single_item_pred_set = prediction_set[
    (prediction_set['item_id'] == item_id) & 
    (prediction_set['dept_id'] == dept_id) & 
    (prediction_set['cat_id'] == cat_id) & 
    (prediction_set['store_id'] == store_id) & 
    (prediction_set['d'].isin(forecast_days))
].copy()




# Create the pivot table for recent sales data
recent_sales_pivot = recent_sales.pivot(index='id', columns='d', values='sales').fillna(0)

recent_sales_pivot

print(f"(recent_sales_pivot ): ")
print(f"len(recent_sales_pivot): {len(recent_sales_pivot)}")
print(f"recent_sales_pivot dataset shape : {recent_sales_pivot.shape}")
print(recent_sales_pivot.head())

# Updated get_dynamic_lag function with added fallback handling
def get_dynamic_lag(row, day, lag):
    """Get the lagged value for a specific row and day, using historical data if available."""
    target_day = f'd_{int(day.split("_")[1]) - lag}'
    
    if target_day in recent_sales_pivot.columns:
       
        return recent_sales_pivot.loc[row['id'], target_day]
    else:
       
        past_index = single_item_pred_set[(single_item_pred_set['id'] == row['id']) & 
                                          (single_item_pred_set['d'] == target_day)].index
        # Return the first available sales value or NaN if past_index is empty
        return single_item_pred_set.loc[past_index, 'sales'].values[0] if len(past_index) > 0 else np.nan
    

single_item_pred_set['sales'] = np.nan


rolling_windows = [7, 28]


for idx, row in single_item_pred_set.iterrows():
    day = row['d']
    single_item_pred_set.loc[idx, 'sales_lag_7'] = get_dynamic_lag(row, day, 7)
    single_item_pred_set.loc[idx, 'sales_lag_28'] = get_dynamic_lag(row, day, 28)


for window in rolling_windows:
    rolling_means = recent_sales.groupby('id')['sales'].apply(
        lambda x: x.iloc[-window:].mean() if len(x) >= window else x.mean()
    ).fillna(1)
    rolling_stds = recent_sales.groupby('id')['sales'].apply(
        lambda x: x.iloc[-window:].std() if len(x) >= window else x.std()
    ).fillna(1)


    single_item_pred_set[f'rolling_sales_mean_{window}'] = single_item_pred_set['id'].map(rolling_means)
    single_item_pred_set[f'rolling_sales_std_{window}'] = single_item_pred_set['id'].map(rolling_stds)
    
single_item_pred_set

# Convert 'd' to string format to allow comparison
single_item_pred_set['d_str'] = single_item_pred_set['d'].astype(str)

# Assuming 'item_id' represents the unique ID for the item being forecasted
item_id = single_item_pred_set['id'].iloc[0]  # Fetch the unique item ID

print(f"(single_item_pred_set ): ")
print(f"len(single_item_pred_set): {len(single_item_pred_set)}")
print(f"single_item_pred_set dataset shape : {single_item_pred_set.shape}")
print(single_item_pred_set.head())
print(single_item_pred_set.info())

# Prepare the features from the prediction set
X_forecast = single_item_pred_set[feature_columns]



# Predict the sales for the next 28 days
forecast_predictions = model.predict(X_forecast)

# Add the predictions to the dataframe
single_item_pred_set['predicted_sales'] = forecast_predictions

print(f"(single_item_pred_set 222): ")
print(f"len(single_item_pred_set): {len(single_item_pred_set)}")
print(f"single_item_pred_set dataset shape : {single_item_pred_set.shape}")
print(single_item_pred_set.head())

# Display the predicted sales for the next 28 days
print(f"Predicted sales for the next 28 days for item {item_id} in store {store_id}:")
print(single_item_pred_set[['d', 'predicted_sales']])




# Create a Plotly figure for visualization
fig = go.Figure()

# Add actual sales for the validation period
fig.add_trace(go.Scatter(
    x=valid_data['d'],  # Using 'd' column for consistent scale
    y=valid_data['sales'],
    mode='lines+markers',
    name='Actual Sales (Validation)',
    line=dict(color='blue')
))

# Add predicted sales for the validation period
fig.add_trace(go.Scatter(
    x=valid_data['d'],
    y=valid_data['predicted_sales'],
    mode='lines+markers',
    name='Predicted Sales (Validation)',
    line=dict(color='orange')
))

# Add actual sales for the forecast period, using `d` column instead of `date`
fig.add_trace(go.Scatter(
    x=single_item_pred_set['d'],  # Using 'd' column for consistency
    y=single_item_pred_set['sales'],
    mode='lines+markers',
    name='Actual Sales (Forecast)',
    line=dict(color='green')
))

# Add predicted sales for the forecast period
fig.add_trace(go.Scatter(
    x=single_item_pred_set['d'],
    y=single_item_pred_set['predicted_sales'],
    mode='lines+markers',
    name='Predicted Sales (Forecast)',
    line=dict(color='red')
))

# Update layout for enhanced visualization
fig.update_layout(
    title=f"Actual vs Predicted Sales (Validation and Forecast) for (item_id={item_id}, dept_id={dept_id}, cat_id={cat_id}, store_id={store_id})",
    xaxis_title="Day",
    yaxis_title="Sales",
    legend_title="Legend",
    template="plotly_dark"
)

# Show the plot
fig.write_image("sales_comparison_plot for prediction set_last_2_year.png")



prediction_set
