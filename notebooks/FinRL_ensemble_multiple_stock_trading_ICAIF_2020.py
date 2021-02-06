#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading Using Ensemble Strategy
#
# Tutorials to use OpenAI DRL to trade multiple stocks using ensemble strategy in one Jupyter Notebook | Presented at ICAIF 2020
#
# * This notebook is the reimplementation of our paper: Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, using FinRL.
# * Check out medium blog for detailed explanations: https://medium.com/@ai4finance/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02
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


# ## install finrl library
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

# In[2]:


# matplotlib.use('Agg')
import datetime
import itertools
import os
import sys
import warnings
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.config import config
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.model.models import DRLAgent, DRLEnsembleAgent
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.trade.backtest import BackTestPlot, BackTestStats, BaselineStats

warnings.filterwarnings("ignore")


# In[3]:


get_ipython().run_line_magic("matplotlib", "inline")


sys.path.append("../FinRL-Library")


# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[4]:


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

# In[5]:


# from config.py start_date is a string
config.START_DATE


# In[6]:


print(config.SRI_KEHATI_TICKER)


# In[7]:


df = YahooDownloader(
    start_date=config.START_DATE,
    end_date="2021-01-19",
    ticker_list=config.SRI_KEHATI_TICKER,
).fetch_data()


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.shape


# In[11]:


df.sort_values(["date", "tic"]).head()


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# In[12]:


fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df)


# In[13]:


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


# In[14]:


processed_full.sample(5)


# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
#
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# In[15]:


config.TECHNICAL_INDICATORS_LIST


# In[16]:


stock_dimension = len(processed_full.tic.unique())
state_space = (
    1 + 2 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[17]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 50_000_000
    / 100,  # Since in Indonesia the minimum number of shares per trx is 100, then we scaled the initial amount by dividing it with 100
    "buy_cost_pct": 0.0019,  # IPOT has 0.19% buy cost
    "sell_cost_pct": 0.0029,  # IPOT has 0.29% sell cost
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5,
}


# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.
#
# * In this notebook, we are training and validating 3 agents (A2C, PPO, DDPG) using Rolling-window Ensemble Method ([reference code](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/80415db8fa7b2179df6bd7e81ce4fe8dbf913806/model/models.py#L92))

# In[18]:


rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
train_start = "2000-01-01"
train_end = "2019-01-01"
val_test_start = "2019-01-01"
val_test_end = "2021-01-18"

ensemble_agent = DRLEnsembleAgent(
    df=processed_full,
    train_period=(train_start, train_end),
    val_test_period=(val_test_start, val_test_end),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)


# In[19]:


A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0005}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_model_kwargs = {
    "action_noise": "ornstein_uhlenbeck",
    "buffer_size": 50_000,
    "learning_rate": 0.000005,
    "batch_size": 128,
}

timesteps_dict = {"a2c": 100_000, "ppo": 100_000, "ddpg": 50_000}


# In[ ]:


df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
)


# In[30]:


df_summary


# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# In[31]:


unique_trade_date = processed_full[
    (processed_full.date > val_test_start) & (processed_full.date <= val_test_end)
].date.unique()


# In[32]:


df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

df_account_value = pd.DataFrame()
for i in range(
    rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window
):
    temp = pd.read_csv("results/account_value_trade_{}_{}.csv".format("ensemble", i))
    df_account_value = df_account_value.append(temp, ignore_index=True)
sharpe = (
    (252 ** 0.5)
    * df_account_value.account_value.pct_change(1).mean()
    / df_account_value.account_value.pct_change(1).std()
)
print("Sharpe Ratio: ", sharpe)
df_account_value = df_account_value.join(
    df_trade_date[validation_window:].reset_index(drop=True)
)


# In[33]:


df_account_value.head()


# In[34]:


get_ipython().run_line_magic("matplotlib", "inline")
df_account_value.account_value.plot()


# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# In[35]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = BackTestStats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)


# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[36]:


print("==============Compare to IHSG===========")
get_ipython().run_line_magic("matplotlib", "inline")
BackTestPlot(
    df_account_value,
    baseline_ticker="^JKSE",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)


# <a id='6.3'></a>
# ## 7.3 Baseline Stats

# In[37]:


print("==============Get Baseline Stats===========")
baseline_perf_stats = BaselineStats(
    "^JKSE",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)