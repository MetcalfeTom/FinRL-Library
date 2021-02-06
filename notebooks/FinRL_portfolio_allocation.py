#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_portfolio_allocation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Deep Reinforcement Learning for Stock Trading from Scratch: Portfolio Allocation
#
# Tutorials to use OpenAI DRL to perform portfolio allocation in one Jupyter Notebook | Presented at NeurIPS 2020: Deep RL Workshop
#
# * This blog is based on our paper: FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, presented at NeurIPS 2020: Deep RL Workshop.
# * Check out medium blog for detailed explanations:
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

# In[1]:


## install finrl library
get_ipython().system(
    "pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git"
)


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

# In[3]:


import datetime
import os
import sys

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

import pyfolio
from finrl.config import config
from finrl.env.env_portfolio import StockPortfolioEnv
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.model.models import DRLAgent
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.trade.backtest import (
    BackTestPlot,
    BackTestStats,
    BaselineStats,
    backtest_strat,
    baseline_strat,
)
from pyfolio import timeseries

matplotlib.use("Agg")


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

# In[5]:


print(config.DOW_30_TICKER)


# In[7]:


df = YahooDownloader(
    start_date="2008-01-01", end_date="2021-01-01", ticker_list=config.DOW_30_TICKER
).fetch_data()


# In[ ]:


df.head()


# In[ ]:


df.shape


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# In[8]:


fe = FeatureEngineer(
    use_technical_indicator=True, use_turbulence=False, user_defined_feature=False
)

df = fe.preprocess_data(df)


# In[9]:


df.shape


# In[10]:


df.head()


# ## Add covariance matrix as states

# In[11]:


# add covariance matrix as states
df = df.sort_values(["date", "tic"], ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
# look back is one year
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i - lookback : i, :]
    price_lookback = data_lookback.pivot_table(
        index="date", columns="tic", values="close"
    )
    return_lookback = price_lookback.pct_change().dropna()
    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame({"date": df.date.unique()[lookback:], "cov_list": cov_list})
df = df.merge(df_cov, on="date")
df = df.sort_values(["date", "tic"]).reset_index(drop=True)


# In[12]:


df.shape


# In[13]:


df.head()


# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
#
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# ## Training data split: 2009-01-01 to 2018-12-31

# In[14]:


train = data_split(df, "2009-01-01", "2019-01-01")
# trade = data_split(df, '2020-01-01', config.END_DATE)


# In[ ]:


train.head()


# ## Environment for Portfolio Allocation
#

# In[23]:


matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# In[24]:


stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[25]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)


# In[26]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# In[27]:


# initialize
agent = DRLAgent(env=env_train)


# ### Model 1: **A2C**
#

# In[28]:


agent = DRLAgent(env=env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)


# In[29]:


trained_a2c = agent.train_model(
    model=model_a2c, tb_log_name="a2c", total_timesteps=50000
)


# ### Model 2: **PPO**
#

# In[ ]:


agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)


# In[ ]:


trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps=80000
)


# ### Model 3: **DDPG**
#

# In[ ]:


agent = DRLAgent(env=env_train)
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}


model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)


# In[ ]:


trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
)


# ### Model 4: **SAC**
#

# In[ ]:


agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)


# In[ ]:


trained_sac = agent.train_model(
    model=model_sac, tb_log_name="sac", total_timesteps=50000
)


# ## Trading
# Assume that we have $1,000,000 initial capital at 2019-01-01. We use the DDPG model to trade Dow jones 30 stocks.

# In[30]:


trade = data_split(df, "2019-01-01", "2021-01-01")
# trade = data_split(df,config.START_TRADE_DATE,config.END_DATE)


# In[31]:


trade.shape


# In[32]:


trade.head()


# In[33]:


e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=trained_a2c, test_data=trade, test_env=env_trade, test_obs=obs_trade
)


# In[34]:


df_daily_return.head()


# In[35]:


df_actions.head()


# In[ ]:


df_actions.to_csv("df_actions.csv")


# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# In[36]:


DRL_strat = backtest_strat(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=DRL_strat,
    factor_returns=DRL_strat,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)


# In[37]:


print("==============DRL Strategy Stats===========")
perf_stats_all


# In[43]:


print("==============Get Index Stats===========")
baesline_perf_stats = BaselineStats(
    "^DJI", baseline_start="2019-01-01", baseline_end="2021-01-01"
)


# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[41]:


dji, dow_strat = baseline_strat("^DJI", "2019-01-01", "2021-01-01")


# In[42]:


get_ipython().run_line_magic("matplotlib", "inline")
with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(
        returns=DRL_strat, benchmark_rets=dow_strat, set_context=False
    )


# In[ ]:
