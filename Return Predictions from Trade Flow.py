#!/usr/bin/env python
# coding: utf-8

# # Return Predictions From Trade Flow

# # 1. Introduction:

# This work is assessing the trade flow to generate profit opportunities in three cryptotoken markets. Trade flow is a running tally of signed trade sizes where the sign is defined as 1 if the trade is seller-initiated and -1 if it was buyer-initiated. All reported trades within the last time period of length tau are examined for the profit-generating purpose. Three crypto token pairs considered for the analysis are ETH-BTC, BTC-USD and ETH-USD.

# Import all the python packages needed:

# In[1]:


import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import plotnine as p9
import functools
import itertools
import random
import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from scipy.stats import probplot
from datetime import datetime
from io import BytesIO
from io import StringIO
from zipfile import ZipFile
from pandas.plotting import register_matplotlib_converters
from plotnine import ggplot, aes, geom_line, labs, scale_x_date, theme_minimal, element_text, theme, geom_ribbon, geom_point
from mizani.breaks import date_breaks
from mizani.formatters import date_format
register_matplotlib_converters()


# # 2. Various functions that is to be used for the analysis:

# # Trade Flow Calculation Function (tau interval):

# In[2]:


def trade_flow(df, time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', tau = '10s'):
    """
    Calculate the trade flow based on the cumulative size within a given time interval (tau).

    Parameters:
    - df: DataFrame containing the trade data.
    - time_col: The name of the column containing the timestamp in nanoseconds.
    - size_col: The name of the column containing the trade sizes.
    - sign_col: The name of the column containing the trade side (buy/sell indicator).
    - tau: The time interval as a string, e.g. '10s' for 10 seconds.

    Returns:
    - DataFrame with a new 'trade_flow' column representing the trade flow.
    """
    df = df.copy()
    
    # Creating the signed size column
    df['cum_size_billionths'] = df[size_col] * np.sign(df[sign_col])
    
    # Convert the timestamp to a datetime index
    df['timestamp'] = pd.to_datetime(df[time_col], unit='ns')

    # Ensure the data is sorted by timestamp
    df.sort_values(by=time_col, inplace=True)

    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)

    # Calculate the trade flow using a rolling window and sum within the interval tau
    df['trade_flow'] = df['cum_size_billionths'].rolling(tau, closed='left').sum()
    
    df['trade_flow'] = df['trade_flow']/ 1e9
    
    # Reset the index to remove the timestamp from the index position
    df.reset_index(drop=True, inplace=True)
    
    return df


# # T-second Forward Return Calculation Function:

# In[3]:


def T_second_forward_return(trades, book, trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', time_col = 'timestamp_utc_nanoseconds', T = '5s'):
    """
    Calculate the T-second forward return within a given forward time interval (T).

    Parameters:
    - df1: DataFrame containing the trade data.
    - df2: DataFrame containing the book data.
    - trade_price: The name of the column containing trade price at time t.
    - mid_price_tplusT: The name of the column containing trade price at time t+T.
    - time_col: The name of the column containing the timestamp in nanoseconds.
    - T: The forward time interval as a string, e.g. '5s' for 5 seconds.

    Returns:
    - DataFrame with a new 'T_second_forward_return' column representing the trade flow.
    """
    # Create a temporary 'timestamp' column for both trades and book data
    trades['timestamp'] = pd.to_datetime(trades[time_col], unit='ns')
    book['timestamp'] = pd.to_datetime(book[time_col], unit='ns')
    
    # Set the temporary 'timestamp' column as the index for both DataFrames
    trades.set_index('timestamp', inplace=True)
    book.set_index('timestamp', inplace=True)

    # Forward fill the book data to the timestamps of the trades data plus T seconds
    later = book.reindex(trades.index + pd.to_timedelta(T), method='ffill')
    later.index = trades.index

    # Calculate the T-second forward return
    trades['T_second_forward_return'] = (later[mid_price_col] / trades[trade_price_col]) - 1

    # Reset the index to convert the datetime index back into a column and drop the temporary 'timestamp' column
    trades.reset_index(drop=True, inplace=True)
    
    # Drop the temporary 'timestamp' column from the book DataFrame as well
    book.reset_index(drop=True, inplace=True)
    
    return trades
    
        


# # Function to create the training and test split datasets:

# In[4]:


def train_test_data(trades, train_size=0.4):
    """
    Split the trades dataset into training and testing sets.

    Parameters:
    - trades: DataFrame containing the trade data.
    - train_size: The proportion of the dataset to include in the train split.

    Returns:
    - A tuple containing the training set and the testing set.
    """
    # Calculate the index to split on
    split_index = int(len(trades) * train_size)
    
    # Split the DataFrame into training and testing sets
    train_set = trades[:split_index]
    test_set = trades[split_index:]
    
    return train_set, test_set


# # Regression function to find coefficient Beta of regression:

# In[5]:


def regression(train_data, test_data, flow_col='trade_flow', return_col='T_second_forward_return'):
    """
    Fit a regression model without an intercept and predict returns on the test set.
    """
    # Ensure there are no NaN values in the columns used for regression
    train_data = train_data.dropna(subset=[flow_col, return_col])
    test_data = test_data.dropna(subset=[flow_col])
    
    # Fit regression model on the training set without a constant
    model = OLS(train_data[return_col], train_data[flow_col]).fit()
    
    # Get the coefficient of regression
    beta = model.params[flow_col]
    
    # Predict returns on the test set
    test_data['predicted_return'] = model.predict(test_data[flow_col])
    
    # Calculate the R-squared value
    r_squared = model.rsquared_adj
    
    return beta, test_data, r_squared


# # Function to find trade Signals:

# In[6]:


def find_trade_signals(test_data, predicted_return_col='predicted_return', j=0.000001):
    """
    Find trade signals based on the threshold j.
    """
    # Identify potential trade signals where the absolute predicted return is greater than the threshold j
    test_data['trade_signal'] = test_data[predicted_return_col].abs() > j
    
    return test_data


# # Function to create PnL (without trading costs) dataframe:

# In[7]:


def calculate_pnl(test_pred, book, T, position_size=0.05):
    """
    Calculate and accumulate P&L for trade signals within interval T.

    Parameters:
    - test_pred: DataFrame containing test predictions with trade signals.
    - book: DataFrame containing book data with mid prices.
    - T: Time interval in seconds as a string, e.g., '5s'.
    - position_size: Multiplier to determine the position size.

    Returns:
    - pnl_df: DataFrame containing P&L information.
    """
    # Filter for trade signals
    signals = test_pred[test_pred['trade_signal']]

    # Merge book data to get the mid price at the signal's timestamp
    signals = signals.merge(book[['timestamp_utc_nanoseconds', 'Mid']], on='timestamp_utc_nanoseconds', how='left')
           
    signals['position'] = position_size * signals['SizeBillionths'] * signals['Side'].apply(np.sign)

    # Get the price at the end of the interval T
    signals['timestamp_end'] = pd.to_datetime(signals['timestamp_utc_nanoseconds'], unit='ns') + pd.to_timedelta(T)
    book['timestamp'] = pd.to_datetime(book['timestamp_utc_nanoseconds'], unit='ns')
    book.set_index('timestamp', inplace=True)
    
    signals['mid_price_end'] = signals['timestamp_end'].apply(lambda x: book.at[x, 'Mid'] if x in book.index else np.nan)

    # Calculate P&L for each T interval
    signals['plT'] = signals['position'] * (signals['mid_price_end'] - signals['PriceMillionths'])

    # Accumulate P&L over all intervals
    signals['acc_qty'] = signals['position'].cumsum().shift(fill_value = 0)
    signals['acc_plT'] = signals['plT'].cumsum().shift(fill_value=0)

    # Create the P&L DataFrame
    pnl_df = signals[['timestamp_utc_nanoseconds', 'position', 'PriceMillionths', 'Mid', 'acc_qty', 'plT', 'acc_plT']].copy()

    pnl_df['Mid'].fillna(method='ffill', inplace=True)
    
    # Convert 'PriceMillionths' to actual price
    pnl_df['PriceMillionths'] = pnl_df['PriceMillionths'] / 1e6

    # Assuming you already have a 'position' column calculated based on the original size
    # and you want to adjust it using the actual size
    pnl_df['position'] = pnl_df['position'] / 1e9

    pnl_df['Mid'] = pnl_df['Mid'] / 1e6  
    pnl_df['plT'] = pnl_df['position'] * (pnl_df['Mid'] - pnl_df['PriceMillionths'])
    pnl_df['acc_plT'] = pnl_df['plT'].cumsum().shift(fill_value=0)

    
    return pnl_df


# # Function to create PnL (with trading costs) dataframe:

# Trading costs for the pair has been assumed to be 0.2%.

# In[8]:


def calculate_pnl_with_tc(test_pred, book, T, position_size=0.05):
    """
    Calculate and accumulate P&L for trade signals within interval T.

    Parameters:
    - test_pred: DataFrame containing test predictions with trade signals.
    - book: DataFrame containing book data with mid prices.
    - T: Time interval in seconds as a string, e.g., '5s'.
    - position_size: Multiplier to determine the position size.

    Returns:
    - pnl_df: DataFrame containing P&L information.
    """
    # Filter for trade signals
    signals = test_pred[test_pred['trade_signal']]

    # Merge book data to get the mid price at the signal's timestamp
    signals = signals.merge(book[['timestamp_utc_nanoseconds', 'Mid']], on='timestamp_utc_nanoseconds', how='left')
           
    signals['position'] = position_size * signals['SizeBillionths'] * signals['Side'].apply(np.sign)

    # Get the price at the end of the interval T
    signals['timestamp_end'] = pd.to_datetime(signals['timestamp_utc_nanoseconds'], unit='ns') + pd.to_timedelta(T)
    book['timestamp'] = pd.to_datetime(book['timestamp_utc_nanoseconds'], unit='ns')
    book.set_index('timestamp', inplace=True)
    
    signals['mid_price_end'] = signals['timestamp_end'].apply(lambda x: book.at[x, 'Mid'] if x in book.index else np.nan)

    # Calculate P&L for each T interval
    signals['plT'] = signals['position'] * (signals['mid_price_end'] - signals['PriceMillionths'])

    # Accumulate P&L over all intervals
    signals['acc_qty'] = signals['position'].cumsum().shift(fill_value = 0)
    signals['acc_plT'] = signals['plT'].cumsum().shift(fill_value=0)

    # Create the P&L DataFrame
    pnl_df = signals[['timestamp_utc_nanoseconds', 'position', 'PriceMillionths', 'Mid', 'acc_qty', 'plT', 'acc_plT']].copy()

    pnl_df['Mid'].fillna(method='ffill', inplace=True)
    
    # Convert 'PriceMillionths' to actual price
    pnl_df['PriceMillionths'] = pnl_df['PriceMillionths'] / 1e6

    # Assuming you already have a 'position' column calculated based on the original size
    # and you want to adjust it using the actual size
    pnl_df['position'] = pnl_df['position'] / 1e9

    pnl_df['Mid'] = pnl_df['Mid'] / 1e6  
    pnl_df['plT'] = pnl_df['position'] * (pnl_df['Mid'] - pnl_df['PriceMillionths'])*0.998
    pnl_df['acc_plT'] = pnl_df['plT'].cumsum().shift(fill_value=0)

    
    return pnl_df


# # Function to calculate Sharpe ratio, Max Drawdowns, Downside Deviations and Cumulative PnL plot:

# In[9]:


def calculate_pnl_stats(pnl, T):
    # Set the timestamp as the index
    pnl['timestamp'] = pd.to_datetime(pnl['timestamp_utc_nanoseconds'], unit='ns')
    pnl.set_index('timestamp', inplace=True)

    # Group and sum P&L into T-second bins
    grouped_plT = pnl['plT'].resample(T).sum()

    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    sharpe_ratio = grouped_plT.mean() / grouped_plT.std() if grouped_plT.std() != 0 else np.nan

    # Calculate maximum drawdown
    running_max = np.maximum.accumulate(grouped_plT.cumsum())
    drawdown = running_max - grouped_plT.cumsum()
    max_drawdown = drawdown.max()

    # Calculate downside deviations (using 0 as the minimum acceptable return)
    downside_deviations = grouped_plT[grouped_plT < 0].std() if not grouped_plT[grouped_plT < 0].empty else np.nan

    # Plot cumulative P&L over time
    cumulative_plT = grouped_plT.cumsum()
    cumulative_plT.plot()
    plt.title('Cumulative P&L over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative P&L')
    plt.show()
    
    pnl.reset_index(drop=True, inplace=True)
    
    return sharpe_ratio, max_drawdown, downside_deviations, cumulative_plT, grouped_plT



# # Reliability and Stability of beta:

# In[10]:


def analyze_beta_stability(beta_df):
    """
    Analyze the stability of beta values.
    
    Parameters:
    beta_df (DataFrame): DataFrame containing 'tau', 'T', and 'beta' columns.

    Returns:
    dict: Dictionary containing key metrics for beta stability.
    """
    # Coefficient of Variation
    cv = beta_df['beta'].std() / beta_df['beta'].mean()

    # Standard Deviation
    beta_std = beta_df['beta'].std()

    # Range
    beta_range = beta_df['beta'].max() - beta_df['beta'].min()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=beta_df, x='beta')
    plt.title('Box Plot of Beta Values')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.distplot(beta_df['beta'], bins=30)
    plt.title('Distribution of Beta Values')
    plt.show()

    # Return the key metrics
    return {
        'Coefficient of Variation': cv,
        'Beta Standard Deviation': beta_std,
        'Beta Range': beta_range
    }


# # Pipeline to get beta for different tau and T:

# In[11]:


def pipeline(trades, book, time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', tau = '10s', T = '5s', train_size=0.4):
    # Step 1: Calculate trade flow
    trades = trade_flow(trades, time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', tau = tau)
    
    # Step 2: Calculate T-second forward return
    trades = T_second_forward_return(trades, book, trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', time_col = 'timestamp_utc_nanoseconds', T = T)
    
    # Step 3: Split data into train and test
    train, test = train_test_data(trades, train_size = train_size)
    
    # Step 4: Perform regression and get beta
    beta, test_data, r_squared = regression(train, test, flow_col='trade_flow', return_col='T_second_forward_return')
    
    return beta


# # 3. Analysis of ETH-BTC pair:

# Reading the trade and book data for ETH-BTC pair:

# In[12]:


#Reading the trades data ETH-BTC
trades_file_path = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\trades_narrow_ETH-BTC_2023.delim.gz'
trades_ETH_BTC = pd.read_csv(trades_file_path, delimiter='\t')


# In[13]:


trades_ETH_BTC.head()


# In[14]:


trades_ETH_BTC.shape


# In[15]:


#Reading the book data ETH-BTC
book_narrow_file_path = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\book_narrow_ETH-BTC_2023.delim.gz'
book_narrow_ETH_BTC = pd.read_csv(book_narrow_file_path, delimiter='\t')


# In[16]:


book_narrow_ETH_BTC.shape


# In[17]:


book_narrow_ETH_BTC.head()


# Sorting the datasets by timestamps and remove the duplicates by keeping the latest value:

# In[18]:


# Sort by 'timestamp_utc_nanoseconds' in descending order to ensure the latest entries are first
trades_ETH_BTC.sort_values(by='timestamp_utc_nanoseconds', ascending=False, inplace=True)

trades_ETH_BTC.drop_duplicates(subset=['timestamp_utc_nanoseconds'], keep='first', inplace=True)

trades_ETH_BTC.sort_values(by='timestamp_utc_nanoseconds', ascending=True, inplace=True)


# # Calling the trade flow function to calculate trade flow for ETH-BTC pair:

# In[19]:


trades_ETH_BTC = trade_flow(trades_ETH_BTC,  time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', tau='60s')


# # Calculating the T-second forward returns:

# In[20]:


trades_ETH_BTC = T_second_forward_return(trades_ETH_BTC, book_narrow_ETH_BTC, trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', time_col = 'timestamp_utc_nanoseconds', T = '50s')


# In[21]:


trades_ETH_BTC.shape


# In[22]:


trades_ETH_BTC.head()


# # Looking at the data to check for nan values before regression:

# In[23]:


trades_ETH_BTC.info()


# In[24]:


plt.figure(figsize=(10, 6))
plt.hist(trades_ETH_BTC['trade_flow'].dropna(), bins=50, alpha=0.7)
plt.title('Distribution of trade_flow')
plt.xlabel('trade_flow')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Fill the nan values for trade flow by keeping the distribution of trade flow the same:

# In[25]:


# Fill NaN values in 'trade_flow' with the median value
median_flow = trades_ETH_BTC['trade_flow'].median()
trades_ETH_BTC['trade_flow'].fillna(median_flow, inplace=True)


# In[26]:


trades_ETH_BTC.describe()


# Splitting the dataset into training and test data:

# In[27]:


train_ETH_BTC, test_ETH_BTC = train_test_data(trades_ETH_BTC, train_size=0.4)


# In[28]:


train_ETH_BTC.head()


# Checking for percentage for training and test data:

# In[29]:


print(f"Training set length: {len(train_ETH_BTC)}")
print(f"Testing set length: {len(test_ETH_BTC)}")


# # Calling the regression function:

# In[30]:


beta_ETH_BTC, test_pred_ETH_BTC, r2_ETH_BTC = regression(train_ETH_BTC, test_ETH_BTC, flow_col='trade_flow', return_col='T_second_forward_return')


# In[31]:


r2_ETH_BTC


# In[32]:


beta_ETH_BTC


# # Calculate beta for various tau and T pairs:

# In[33]:


tau_T_pairs = [('5s', '1s'), ('10s', '5s'), ('15s', '10s'), ('20s', '15s'), ('25s', '20s'), ('30s', '25s'), ('35s', '30s'), ('40s', '30s'), ('50s', '40s'), ('60s', '50s'), ('100s', '80s')]


# In[34]:


beta_values = {}

# Loop through each pair of tau and T
for tau, T in tau_T_pairs:
    beta = pipeline(trades_ETH_BTC, book_narrow_ETH_BTC, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=tau, T=T, 
                    train_size=0.4)
    # Store the beta value in the dictionary
    beta_values[(tau, T)] = beta


# In[35]:


beta_records = [{'tau': tau, 'T': T, 'beta': beta} for (tau, T), beta in beta_values.items()]

# Create a DataFrame
beta_df = pd.DataFrame(beta_records)

# Display the DataFrame
print(beta_df)


# For various pairs of tau and T beta has been calculated and we can see that beta increase as we increase the trading interval and T-second and then decrease after some value.

# # Looking at the reliability and stability of beta:

# In[36]:


rel_stab_beta = analyze_beta_stability(beta_df)
print(rel_stab_beta)


# # Various beta for different training sizes by considering best tau and T:

# In[37]:


best_tau = '60s'
best_T = '50s'

# Define the range of training sizes
train_sizes = [0.4, 0.5, 0.6, 0.7, 0.8]

# Initialize an empty list to store the results
beta_values = []

# Loop over the training sizes
for train_size in train_sizes:
    beta = pipeline(trades_ETH_BTC, book_narrow_ETH_BTC, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=best_tau, T=best_T, 
                    train_size=train_size)
    beta_values.append(beta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, beta_values, marker='o')
plt.title('Beta Values for Different Training Sizes')
plt.xlabel('Training Size')
plt.ylabel('Beta Value')
plt.xticks(train_sizes)
plt.grid(True)
plt.show()


# We can see from the plot that the beta value increases as we increase the training dataset size.

# # Look at the return prediction from the regression function:

# In[38]:


test_pred_ETH_BTC


# In[39]:


test_pred_ETH_BTC['predicted_return'].describe()


# # Deciding j:

# j has been decided in such a way that trade can be performed 60% of the time:

# In[40]:


# Calculate the absolute values of the predicted returns
absolute_predicted_returns = test_pred_ETH_BTC['predicted_return'].abs()

# Find the value of j such that 70% of the predicted returns are greater than j
j_ETH_BTC = np.percentile(absolute_predicted_returns, 40)  # 40th percentile since we want the lower threshold

j_ETH_BTC


# Finding trade signals for the decided j:

# In[41]:


test_pred_ETH_BTC = find_trade_signals(test_pred_ETH_BTC, predicted_return_col='predicted_return', j=j_ETH_BTC)


# In[42]:


test_pred_ETH_BTC.head()


# In[43]:


test_pred_ETH_BTC['trade_signal'].value_counts()


# # Creating the pnl dataframe:

# In[44]:


pnl_ETH_BTC = calculate_pnl(test_pred_ETH_BTC, book_narrow_ETH_BTC, T = '50s', position_size=0.1)


# In[45]:


pnl_ETH_BTC


# # PnL with trading costs:

# In[46]:


pnl_ETH_BTC_with_tc = calculate_pnl_with_tc(test_pred_ETH_BTC, book_narrow_ETH_BTC, T = '50s', position_size=0.1)


# In[47]:


pnl_ETH_BTC_with_tc


# # Calculating pnl stats:

# In[48]:


T = '50s'  
sharpe_ratio_ETH_BTC, max_drawdown_ETH_BTC, downside_deviations_ETH_BTC, cumulative_plT_ETH_BTC, grouped_plt_ETH_BTC = calculate_pnl_stats(pnl_ETH_BTC, T)


# This graph depicts the cumulative Profit and Loss (P&L) over time. The P&L is positive throughout the observed period, suggesting that the investment or trading strategy has been profitable. 

# In[49]:


sharpe_ratio_ETH_BTC


# In[50]:


max_drawdown_ETH_BTC


# In[51]:


downside_deviations_ETH_BTC


# The Sharpe ratio is 0.012, which is positive but relatively low. This suggests that the excess return of the strategy over the risk-free rate is not very high when taking into account the volatility of the returns.
# 
# Here, the max drawdown is very small, indicating that the largest drop in value from a peak was minimal, which suggests low downside volatility and risk in the context of this strategy.
# 
# The downside deviation is a very small number indicating that the downside risk of the strategy is low. Overall, these statistics indicate that the trading strategy or investment in the ETH-BTC pair has been profitable over the specified period, with a positive (though low) risk-adjusted return, minimal maximum drawdown, and very low downside risk.

# Pnl stats with trading costs:

# In[52]:


T = '50s'  
sharpe_ratio_ETH_BTC_tc, max_drawdown_ETH_BTC_tc, downside_deviations_ETH_BTC_tc, cumulative_plT_ETH_BTC_tc, grouped_plt_ETH_BTC_tc = calculate_pnl_stats(pnl_ETH_BTC_with_tc, T)


# In[53]:


sharpe_ratio_ETH_BTC_tc



# In[54]:


max_drawdown_ETH_BTC_tc



# In[55]:


downside_deviations_ETH_BTC_tc


# From the above plots we can say that the pnl stats hardly change after considering the trading costs.

# # 4. Anaysis of BTC-USD pair:

# Reading the trades and book data:

# In[56]:


#Reading the trades data BTC-USD
trades_file_path1 = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\trades_narrow_BTC-USD_2023.delim.gz'
trades_BTC_USD = pd.read_csv(trades_file_path1, delimiter='\t')


# In[57]:


trades_BTC_USD.head()


# In[58]:


trades_BTC_USD.shape


# In[59]:


#Reading the book data BTC-USD
book_narrow_file_path1 = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\book_narrow_BTC-USD_2023.delim.gz'
book_narrow_BTC_USD = pd.read_csv(book_narrow_file_path1, delimiter='\t')


# In[60]:


book_narrow_BTC_USD.head()


# In[61]:


book_narrow_BTC_USD.shape


# Sorting the datasets by timestamps and remove the duplicates by keeping the latest value:

# In[62]:


# Sort by 'timestamp_utc_nanoseconds' in descending order to ensure the latest entries are first
trades_BTC_USD.sort_values(by='timestamp_utc_nanoseconds', ascending=False, inplace=True)

trades_BTC_USD.drop_duplicates(subset=['timestamp_utc_nanoseconds'], keep='first', inplace=True)

trades_BTC_USD.sort_values(by='timestamp_utc_nanoseconds', ascending=True, inplace=True)


# # Calling the trade flow function to calculate trade flow for BTC-USD pair:

# In[63]:


trades_BTC_USD = trade_flow(trades_BTC_USD,  time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', tau='250s')


# # Calculating the T-second forward returns:

# In[64]:


trades_BTC_USD = T_second_forward_return(trades_BTC_USD, book_narrow_BTC_USD, trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', time_col = 'timestamp_utc_nanoseconds', T = '100s')


# In[65]:


trades_BTC_USD.shape


# In[66]:


trades_BTC_USD.head()


# Looking at the data to check for nan vallues before regression:

# In[67]:


trades_BTC_USD.info()


# In[68]:


plt.figure(figsize=(10, 6))
plt.hist(trades_ETH_BTC['trade_flow'].dropna(), bins=50, alpha=0.7)
plt.title('Distribution of trade_flow')
plt.xlabel('trade_flow')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Fill the nan values for trade flow by keeping the distribution of trade flow the same:

# In[69]:


# Fill NaN values in 'trade_flow' with the median value
median_flow1 = trades_BTC_USD['trade_flow'].median()
trades_BTC_USD['trade_flow'].fillna(median_flow1, inplace=True)


# In[70]:


trades_BTC_USD.describe()


# Splitting the dataset into training and test data:

# In[71]:


train_BTC_USD, test_BTC_USD = train_test_data(trades_BTC_USD, train_size=0.6)


# In[72]:


train_BTC_USD.head()


# Checking for percentage for training and test data:

# In[73]:


print(f"Training set length: {len(train_BTC_USD)}")
print(f"Testing set length: {len(test_BTC_USD)}")


# # Calling the regression function:

# In[74]:


beta_BTC_USD, test_pred_BTC_USD, r2_BTC_USD = regression(train_BTC_USD, test_BTC_USD, flow_col='trade_flow', return_col='T_second_forward_return')


# In[75]:


r2_BTC_USD


# In[76]:


beta_BTC_USD


# # Calculate beta for various tau and T pairs:

# In[77]:


tau_T_pairs1 = [('5s', '1s'), ('10s', '5s'), ('15s', '10s'), ('20s', '15s'), ('25s', '20s'), ('30s', '25s'), ('35s', '30s'), ('40s', '30s'), ('50s', '40s'), ('60s', '50s'), ('100s', '80s')]


# In[78]:


beta_values1 = {}

# Loop through each pair of tau and T
for tau, T in tau_T_pairs1:
    beta1 = pipeline(trades_BTC_USD, book_narrow_BTC_USD, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=tau, T=T, 
                    train_size=0.4)
    # Store the beta value in the dictionary
    beta_values1[(tau, T)] = beta1

beta_records = [{'tau': tau, 'T': T, 'beta': beta1} for (tau, T), beta1 in beta_values1.items()]

# Create a DataFrame
beta_df1 = pd.DataFrame(beta_records)

# Display the DataFrame
print(beta_df1)


# # Looking at the reliability and stability of beta:

# In[79]:


rel_stab_beta1 = analyze_beta_stability(beta_df1)
print(rel_stab_beta1)


# From the plot we can see that, beta doesn't varies that much as we change the tau and T intervals.

# # Various beta for different training sizes by considering best tau and T:

# In[80]:


best_tau1 = '250s'
best_T1 = '100s'

# Define the range of training sizes
train_sizes1 = [0.4, 0.5, 0.6, 0.7, 0.8]

# Initialize an empty list to store the results
beta_values2 = []

# Loop over the training sizes
for train_size in train_sizes1:
    beta = pipeline(trades_BTC_USD, book_narrow_BTC_USD, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=best_tau1, T=best_T1, 
                    train_size=train_size)
    beta_values2.append(beta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes1, beta_values2, marker='o')
plt.title('Beta Values for Different Training Sizes')
plt.xlabel('Training Size')
plt.ylabel('Beta Value')
plt.xticks(train_sizes1)
plt.grid(True)
plt.show()


# # Look at the return prediction from the regression function:

# In[81]:


test_pred_BTC_USD


# In[82]:


test_pred_BTC_USD['predicted_return'].describe()


# # Deciding j:

# j has been decided in such a way that trade can be performed 30% of the time:

# In[83]:


# Calculate the absolute values of the predicted returns
absolute_predicted_returns1 = test_pred_BTC_USD['predicted_return'].abs()

# Find the value of j such that 70% of the predicted returns are greater than j
j_BTC_USD = np.percentile(absolute_predicted_returns1, 70)  # 70th percentile since we want the lower threshold

j_BTC_USD


# Finding trade signals for the decided j:

# In[84]:


test_pred_BTC_USD = find_trade_signals(test_pred_BTC_USD, predicted_return_col='predicted_return', j=j_BTC_USD)


# In[85]:


test_pred_BTC_USD.head()


# In[86]:


test_pred_BTC_USD['trade_signal'].value_counts()


# # Creating the pnl dataframe:

# In[87]:


pnl_BTC_USD = calculate_pnl(test_pred_BTC_USD, book_narrow_BTC_USD, T = '40s', position_size=0.1)

pnl_BTC_USD


# Pnl with trading costs:

# In[88]:


pnl_BTC_USD_with_tc = calculate_pnl_with_tc(test_pred_BTC_USD, book_narrow_BTC_USD, T = '50s', position_size=0.1)

pnl_BTC_USD_with_tc


# # Calculating pnl stats:

# In[89]:


T = '100s'  
sharpe_ratio_BTC_USD, max_drawdown_BTC_USD, downside_deviations_BTC_USD, cumulative_plT_BTC_USD, grouped_plt_BTC_USD = calculate_pnl_stats(pnl_BTC_USD, T)


# This graph represents the cumulative Profit and Loss (P&L) over a certain period and we can see a negative P&L for the time period.

# In[90]:


sharpe_ratio_BTC_USD


# In[91]:


max_drawdown_BTC_USD


# In[92]:


downside_deviations_BTC_USD


# As we have a negative Sharpe ratio of -0.057 which suggests that the investment is performing worse than the risk-free rate when adjusted for volatility. 
# 
# The maximum drawdown value of 1406.753 indicates the largest loss that would have been experienced if an investment was made at the peak value before the largest drop.
# 
# The downside deviation value of 33.566 suggests that there is a level of downside volatility in the returns of the investment.

# In summary, these statistics indicate a trading strategy or investment in BTC against the USD that is currently underperforming, with significant losses (as shown by the cumulative P&L), a negative risk-adjusted return (Sharpe ratio), a significant maximum drawdown, and a measure of downside risk (downside deviation).

# Pnl stats with trading costs:

# In[93]:


T = '100s'  
sharpe_ratio_BTC_USD_tc, max_drawdown_BTC_USD_tc, downside_deviations_BTC_USD_tc, cumulative_plT_BTC_USD_tc, grouped_plt_BTC_USD_tc = calculate_pnl_stats(pnl_BTC_USD_with_tc, T)


# In[94]:


sharpe_ratio_BTC_USD_tc


# In[95]:


max_drawdown_BTC_USD_tc


# In[96]:


downside_deviations_BTC_USD_tc


# # 5. Analysis of ETH-USD pair:

# Reading the trade and book data:

# In[97]:


#Reading the trades data ETH-USD
trades_file_path2 = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\trades_narrow_ETH-USD_2023.delim.gz'
trades_ETH_USD = pd.read_csv(trades_file_path2, delimiter='\t')


# In[98]:


trades_ETH_USD.head()


# In[99]:


trades_ETH_USD.shape


# In[100]:


#Reading the book data ETH-USD
book_narrow_file_path2 = r'C:\Users\nihar\Desktop\QTS\Week 4 Tick Level Data\Assignment 4\book_narrow_ETH-USD_2023.delim.gz'
book_narrow_ETH_USD = pd.read_csv(book_narrow_file_path2, delimiter='\t')


# In[101]:


book_narrow_ETH_USD.head()


# In[102]:


book_narrow_ETH_USD.shape


# Sorting the datasets by timestamps and remove the duplicates by keeping the latest value:

# In[103]:


# Sort by 'timestamp_utc_nanoseconds' in descending order to ensure the latest entries are first
trades_ETH_USD.sort_values(by='timestamp_utc_nanoseconds', ascending=False, inplace=True)

trades_ETH_USD.drop_duplicates(subset=['timestamp_utc_nanoseconds'], keep='first', inplace=True)

trades_ETH_USD.sort_values(by='timestamp_utc_nanoseconds', ascending=True, inplace=True)


# # Calling the trade flow function to calculate trade flow for BTC-USD pair:

# In[104]:


trades_ETH_USD = trade_flow(trades_ETH_USD,  time_col='timestamp_utc_nanoseconds', size_col='SizeBillionths', sign_col='Side', tau='50s')


# # Calculating the T-second forward returns:

# In[105]:


trades_ETH_USD = T_second_forward_return(trades_ETH_USD, book_narrow_ETH_USD, trade_price_col = 'PriceMillionths', mid_price_col = 'Mid', time_col = 'timestamp_utc_nanoseconds', T = '40s')


# In[106]:


trades_ETH_USD.shape


# In[107]:


trades_ETH_USD.head()


# Looking at the data to check for nan vallues before regression:

# In[108]:


trades_ETH_USD.info()


# In[109]:


plt.figure(figsize=(10, 6))
plt.hist(trades_ETH_USD['trade_flow'].dropna(), bins=50, alpha=0.7)
plt.title('Distribution of trade_flow')
plt.xlabel('trade_flow')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Fill the nan values for trade flow by keeping the distribution of trade flow the same:

# In[110]:


# Fill NaN values in 'trade_flow' with the median value
median_flow2 = trades_ETH_USD['trade_flow'].median()
trades_ETH_USD['trade_flow'].fillna(median_flow2, inplace=True)


# In[111]:


trades_ETH_USD.describe()


# Splitting the dataset into training and test data:

# In[112]:


train_ETH_USD, test_ETH_USD = train_test_data(trades_ETH_USD, train_size=0.6)


# In[113]:


train_ETH_USD.head()


# Checking for percentage for training and test data:

# In[114]:


print(f"Training set length: {len(train_ETH_USD)}")
print(f"Testing set length: {len(test_ETH_USD)}")


# # Calling the regression function:

# In[115]:


beta_ETH_USD, test_pred_ETH_USD, r2_ETH_USD = regression(train_ETH_USD, test_ETH_USD, flow_col='trade_flow', return_col='T_second_forward_return')


# In[116]:


r2_ETH_USD


# In[117]:


beta_ETH_USD


# # Calculate beta for various tau and T pairs:

# In[118]:


tau_T_pairs2 = [('5s', '1s'), ('10s', '5s'), ('15s', '10s'), ('20s', '15s'), ('25s', '20s'), ('30s', '25s'), ('35s', '30s'), ('40s', '30s'), ('50s', '40s'), ('60s', '50s'), ('100s', '80s')]

beta_values2 = {}

# Loop through each pair of tau and T
for tau, T in tau_T_pairs2:
    beta2 = pipeline(trades_ETH_USD, book_narrow_ETH_USD, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=tau, T=T, 
                    train_size=0.4)
    # Store the beta value in the dictionary
    beta_values2[(tau, T)] = beta2

beta_records = [{'tau': tau, 'T': T, 'beta': beta1} for (tau, T), beta2 in beta_values2.items()]

# Create a DataFrame
beta_df2 = pd.DataFrame(beta_records)

# Display the DataFrame
print(beta_df2)


# # Looking at the reliability and stability of beta:

# In[119]:


rel_stab_beta2 = analyze_beta_stability(beta_df2)
print(rel_stab_beta2)


# The above plots shows that the beta is constant irrespective of any tau interval or T-seconds. 

# # Various beta for different training sizes by considering best tau and T:

# In[120]:


best_tau2 = '50s'
best_T2 = '40s'

# Define the range of training sizes
train_sizes2 = [0.4, 0.5, 0.6, 0.7, 0.8]

# Initialize an empty list to store the results
beta_values3 = []

# Loop over the training sizes
for train_size in train_sizes2:
    beta = pipeline(trades_ETH_USD, book_narrow_ETH_USD, 
                    time_col='timestamp_utc_nanoseconds',
                    size_col='SizeBillionths', 
                    sign_col='Side', 
                    trade_price_col='PriceMillionths', 
                    mid_price_col='Mid', 
                    tau=best_tau2, T=best_T2, 
                    train_size=train_size)
    beta_values3.append(beta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes2, beta_values3, marker='o')
plt.title('Beta Values for Different Training Sizes')
plt.xlabel('Training Size')
plt.ylabel('Beta Value')
plt.xticks(train_sizes2)
plt.grid(True)
plt.show()


# Beta increases as we increase the training dataset size from 0.4 onwards.

# # Look at the return prediction from the regression function:

# In[121]:


test_pred_ETH_USD


# In[122]:


test_pred_ETH_USD['predicted_return'].describe()


# # Deciding j:

# j has been decided in such a way that trade can be performed 30% of the time:

# In[123]:


# Calculate the absolute values of the predicted returns
absolute_predicted_returns2 = test_pred_ETH_USD['predicted_return'].abs()

# Find the value of j such that 70% of the predicted returns are greater than j
j_ETH_USD = np.percentile(absolute_predicted_returns2, 70)  # 70th percentile since we want the lower threshold

j_ETH_USD


# Finding trade signals for the decided j:

# In[124]:


test_pred_ETH_USD = find_trade_signals(test_pred_ETH_USD, predicted_return_col='predicted_return', j=j_ETH_USD)


# In[125]:


test_pred_ETH_USD.head(20)


# In[126]:


test_pred_ETH_USD['trade_signal'].value_counts()


# # Creating the pnl dataframe:

# In[127]:


pnl_ETH_USD = calculate_pnl(test_pred_ETH_USD, book_narrow_ETH_USD, T = '40s', position_size=0.1)


# In[128]:


pnl_ETH_USD


# Pnl with trading costs:

# In[129]:


pnl_ETH_USD_with_tc = calculate_pnl_with_tc(test_pred_ETH_USD, book_narrow_ETH_USD, T = '40s', position_size=0.1)

pnl_ETH_USD_with_tc


# # Calculating pnl stats:

# In[130]:


T = '40s'  
sharpe_ratio_ETH_USD, max_drawdown_ETH_USD, downside_deviations_ETH_USD, cumulative_plT_ETH_USD, grouped_plt_ETH_USD = calculate_pnl_stats(pnl_ETH_USD, T)


# This graph represents the cumulative Profit and Loss (P&L) over a certain period, likely for a trading strategy. A negative P&L, as shown in the graph, indicates a loss over time. 

# In[131]:


sharpe_ratio_ETH_USD


# In[132]:


max_drawdown_ETH_USD


# In[133]:


downside_deviations_ETH_USD


# As we have a negative Sharpe ratio of -0.14059953386636874 which suggests that the investment is performing worse than the risk-free rate when adjusted for volatility. 
# 
# The maximum drawdown value of 188.3267417687262 indicates the largest loss that would have been experienced if an investment was made at the peak value before the largest drop.
# 
# The downside deviation value of 0.6374048419366247 suggests that there is a level of downside volatility in the returns of the investment.

# In summary, these statistics indicate a trading strategy or investment in Ethereum against the USD that is currently underperforming, with significant losses (as shown by the cumulative P&L), a negative risk-adjusted return (Sharpe ratio), a significant maximum drawdown, and a measure of downside risk (downside deviation).

# Pnl stats with trading costs:

# In[134]:


T = '100s'  
sharpe_ratio_ETH_USD_tc, max_drawdown_ETH_USD_tc, downside_deviations_ETH_USD_tc, cumulative_plT_ETH_USD_tc, grouped_plt_ETH_USD_tc = calculate_pnl_stats(pnl_ETH_USD_with_tc, T)


# In[135]:


sharpe_ratio_ETH_USD_tc



# In[136]:


max_drawdown_ETH_USD_tc



# In[137]:


downside_deviations_ETH_USD_tc


# # 6. Conclusion:

# The analysis has been done for three cryptocurrency pairs and we can see that the strategy for ETH-BTC pair is relatively profitable as compared to BTC-USD and ETH-USD and this can be seen from the sharpe ratio, max drawdown, downside deviation and cumulative pnl plots.
# 
# Trade Flow as a Predictor:
# The use of trade flow as a predictor for future returns has shown varying degrees of efficacy across the different cryptocurrency pairs. While the methodology for calculating trade flow and predicting forward returns is sound, the actual predictive power, as evidenced by the calculated beta values, varies depending on the pair and the chosen intervals (tau and T).
# 
# Beta Stability:
# The stability of the beta coefficient varied with different tau and T intervals, suggesting that the relationship between trade flow and future returns is not consistent across different time scales. This implies that the predictive power of trade flow may be more complex than initially assumed and may be influenced by external factors not accounted for in the model.
# 
# Training Size Impact:
# The size of the training set had a noticeable impact on the beta values, indicating that the amount of historical data used to train the model can influence the predictions. As the training size increased, there was a tendency for beta to increase as well, suggesting that having more data points can potentially lead to stronger predictions. However, this did not necessarily translate into better performance metrics, indicating that more data does not automatically mean more profitable trading signals.
# 
# Trading Signal Generation:
# The strategy of determining trade signals based on a certain percentile of predicted returns (with j values chosen to activate trading 60% and 30% of the time for the respective pairs) did not lead to consistently profitable outcomes. This could suggest that the threshold for generating signals may need further optimization, or that other factors may need to be considered when deciding when to trade.
# 
# Transaction Costs Consideration:
# Incorporating transaction costs into the P&L calculations showed that the profitability of trades was further diminished but not significantly when such costs were accounted for.
# 
# In conclusion, while the analytical approach and the developed models have potential, the current implementation does not yield profitable trading strategies for the cryptocurrency pairs analyzed. The analysis highlights the need for further refinement of the trade flow-based predictive model, perhaps by incorporating additional market factors, optimizing the model parameters, or employing more sophisticated risk management techniques. Future research could also explore different timeframes, alternative predictive indicators, and the use of machine learning models to improve the accuracy and profitability of the trade signals.
