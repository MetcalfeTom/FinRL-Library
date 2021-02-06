#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading
#
# Tutorials to use OpenAI DRL to trade multiple stocks in one Jupyter Notebook | Presented at NeurIPS 2020: Deep RL Workshop
#
# * This blog uses FinRL to reproduce the paper: Practical Deep Reinforcement Learning Approach for Stock Trading, Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.
# * Check out medium blog for detailed explanations: https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530
# * Please report any issues to our Github: https://github.com/AI4Finance-LLC/FinRL-Library/issues
# * **Pytorch Version**
#
#

# # Content

# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)
#     * [5.3. Initialize Environment](#4.3)
# * [6.Implement DRL Algorithms](#5)
# * [7.Backtesting Performance](#6)
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)
#     * [7.3. Baseline Stats](#6.3)
#     * [7.3. Compare to Stock Market Index](#6.4)

# <a id='0'></a>
# # Part 1. Problem Definition

# This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
#
# The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:
#
#
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
# an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
#
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
#
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
#
# * Environment: Dow 30 consituents
#
#
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
#

# <a id='1'></a>
# # Part 2. Getting Started- Load Python Packages

# <a id='1.1'></a>
# ## 2.1. Install all the packages through FinRL library
#

# In[ ]:


## install finrl library
# !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git


#
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them.
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# <a id='1.3'></a>
# ## 2.3. Import Packages

# In[ ]:


# matplotlib.use('Agg')
import datetime
import itertools
import os
import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.config import config
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.model.models import DRLAgent
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.trade.backtest import (backtest_plot, backtest_stats, get_baseline,
                                  get_daily_return)

get_ipython().run_line_magic("matplotlib", "inline")



sys.path.append("../FinRL-Library")



# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[ ]:



if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
#

#
#
# -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
#
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
#
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API
#

# In[ ]:


# from config.py start_date is a string
config.START_DATE


# In[ ]:


# from config.py end_date is a string
config.END_DATE


# In[ ]:


print(config.DOW_30_TICKER)


# In[ ]:


df = YahooDownloader(
    start_date="2009-01-01", end_date="2021-01-01", ticker_list=config.DOW_30_TICKER
).fetch_data()


# In[ ]:


df.shape


# In[ ]:


df.sort_values(["date", "tic"], ignore_index=True).head()


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# In[ ]:


fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df)


# In[ ]:


list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])

processed_full = processed_full.fillna(0)


# In[ ]:


processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)


# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
#
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# ## Training data split: 2009-01-01 to 2018-12-31
# ## Trade data split: 2019-01-01 to 2020-09-30

# In[ ]:


train = data_split(processed_full, "2009-01-01", "2019-01-01")
trade = data_split(processed_full, "2019-01-01", "2021-01-01")
print(len(train))
print(len(trade))


# In[ ]:


train.head()


# In[ ]:


trade.head()


# In[ ]:


config.TECHNICAL_INDICATORS_LIST


# In[ ]:


stock_dimension = len(train.tic.unique())
state_space = (
    1 + 2 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[ ]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)


# ## Environment for Training
#
#

# In[ ]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# In[ ]:


agent = DRLAgent(env=env_train)


# ### Model Training: 5 models, A2C DDPG, PPO, TD3, SAC
#

# ### Model 1: A2C
#

# In[ ]:


agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")


# In[ ]:


trained_a2c = agent.train_model(
    model=model_a2c, tb_log_name="a2c", total_timesteps=100000
)


# ### Model 2: DDPG

# In[ ]:


agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")


# In[ ]:


trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
)


# ### Model 3: PPO

# In[ ]:


agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)


# In[ ]:


trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps=100000
)


# ### Model 4: TD3

# In[ ]:


agent = DRLAgent(env=env_train)
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)


# In[ ]:


trained_td3 = agent.train_model(
    model=model_td3, tb_log_name="td3", total_timesteps=30000
)


# ### Model 5: SAC

# In[ ]:


agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)


# In[ ]:


trained_sac = agent.train_model(
    model=model_sac, tb_log_name="sac", total_timesteps=80000
)


# ## Trading
# Assume that we have $1,000,000 initial capital at 2019-01-01. We use the DDPG model to trade Dow jones 30 stocks.

# ### Set turbulence threshold
# Set the turbulence threshold to be greater than the maximum of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile

# In[ ]:


data_turbulence = processed_full[
    (processed_full.date < "2019-01-01") & (processed_full.date >= "2009-01-01")
]
insample_turbulence = data_turbulence.drop_duplicates(subset=["date"])


# In[ ]:


insample_turbulence.turbulence.describe()


# In[ ]:


turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)


# In[ ]:


turbulence_threshold


# ### Trade
#
# DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use the in-sample data from 2009-01 to 2018-12 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# In[ ]:


trade = data_split(processed_full, "2019-01-01", "2021-01-01")
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=380, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model_sac, environment=e_trade_gym
)


# In[ ]:


df_account_value.shape


# In[ ]:


df_account_value.head()


# In[ ]:


df_actions.head()


# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# In[ ]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")


# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[ ]:


print("==============Compare to DJIA===========")
get_ipython().run_line_magic("matplotlib", "inline")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(
    df_account_value,
    baseline_ticker="^DJI",
    baseline_start="2019-01-01",
    baseline_end="2021-01-01",
)


# <a id='6.3'></a>
# ## 7.3 Baseline Stats

# In[ ]:


print("==============Get Baseline Stats===========")
baseline_df = get_baseline(ticker="^DJI", start="2019-01-01", end="2021-01-01")

stats = backtest_stats(baseline_df, value_col_name="close")


# In[ ]:
