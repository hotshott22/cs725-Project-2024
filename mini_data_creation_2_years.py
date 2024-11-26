import pickle
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import warnings  # To suppress warnings
import random  # For generating random numbers

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Function to set a fixed random seed for reproducibility
def seed_everything(seed):
    np.random.seed(seed)  # Set numpy random seed
    random.seed(seed)  # Set built-in random seed

seed_everything(seed=2024)  # Set the seed to 2024

# Load datasets
print("Loading datasets...")
calendar = pd.read_csv("calendar.csv")  # Load calendar dataset
sales_train_evaluation = pd.read_csv("sales_train_evaluation.csv")  # Sales data
sell_prices = pd.read_csv("sell_prices.csv")  # Prices data
print("Datasets loaded successfully.")

# Memory optimization function
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

# Apply memory optimization
print("Optimizing memory usage for datasets...")
calendar = reduce_mem_usage(calendar)
sell_prices = reduce_mem_usage(sell_prices)
sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)

# Step 1: Filter the data for the last 2 years
print("Filtering last 2 years of data...")

# Get the last 2 years of unique days from the calendar dataset
# Assuming the calendar dataset has daily data and is ordered by date
last_2_years = calendar['date'].unique()[-730:]  # 730 days for 2 years

# Filter calendar data for the last 2 years
calendar_mini = calendar[calendar['date'].isin(last_2_years)].copy()

# Filter sales_train_evaluation for the last 2 years
# We need to filter the columns d_XXXX that correspond to the last 730 days
# These columns are 'd_XXXX' where XXXX are day indices
sales_train_evaluation_mini = sales_train_evaluation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id','state_id'] + [f'd_{i}' for i in range(1942 - 730, 1942)]].copy()

# Filter sell_prices for the last 2 years
sell_prices_mini = sell_prices[sell_prices['wm_yr_wk'].isin(calendar_mini['wm_yr_wk'])].copy()

print(f"Filtered data for the last 2 years.")

# Print the first few rows of the original and reduced datasets
print("\nFirst few rows of the original calendar dataset:")
print(calendar.head())

print("\nFirst few rows of the reduced calendar dataset:")
print(calendar_mini.head())

print("\nFirst few rows of the original sales_train_evaluation dataset:")
print(sales_train_evaluation.head())

print("\nFirst few rows of the reduced sales_train_evaluation dataset:")
print(sales_train_evaluation_mini.head())

print("\nFirst few rows of the original sell_prices dataset:")
print(sell_prices.head())

print("\nFirst few rows of the reduced sell_prices dataset:")
print(sell_prices_mini.head())

# Step 2: Save the mini datasets with the prefix 'name_mini'
calendar_mini.to_csv('calendar_mini_2_year.csv', index=False)
sales_train_evaluation_mini.to_csv('sales_train_evaluation_mini_2_year.csv', index=False)
sell_prices_mini.to_csv('sell_prices_mini_2_year.csv', index=False)

print("Mini datasets for last 2 years created and saved as CSV files.")

# Step 3: Forecast for the next 10 days (placeholders for now)
# Add a column for forecasted values for the next 10 days (placeholders for now)
forecast_days = 10  # Forecasting the next 10 days
sales_train_evaluation_mini['forecast_10_days'] = np.nan  # Placeholder for future forecasted values

# Example: Store the forecasted 10-day values in the last columns of the dataset
# In a real scenario, you would train a model and predict values for the next 10 days
# Placeholder forecast values for the next 10 days.
sales_train_evaluation_mini['forecast_10_days'] = np.nan  # Replace this with actual forecast values after training your model

# Save the dataset with forecasted values
sales_train_evaluation_mini.to_csv('sales_train_evaluation_mini_2_years_with_forecast.csv', index=False)

# Save the calendar and sell_prices with the prefix 'name_mini'
calendar_mini.to_csv('calendar_mini_2_years_with_forecast.csv', index=False)
sell_prices_mini.to_csv('sell_prices_mini_2_years_with_forecast.csv', index=False)

# # Step 4: Optionally, save the datasets as pickle files
# with open('calendar_mini_2_years.pkl', 'wb') as f:
#     pickle.dump(calendar_mini, f)

# with open('sales_train_evaluation_mini_2_years.pkl', 'wb') as f:
#     pickle.dump(sales_train_evaluation_mini, f)

# with open('sell_prices_mini_2_years.pkl', 'wb') as f:
#     pickle.dump(sell_prices_mini, f)

print("Datasets for the last 2 years with forecasts for the next 10 days saved successfully.")
