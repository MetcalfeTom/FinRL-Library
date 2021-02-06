#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/Crypto_Binance_Historical_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Fetch historical data

# Python script to fetch historical data from binance using ccxt

# In[1]:


# Install openpyxl and CCXT
get_ipython().system("pip install openpyxl ccxt")


# In[2]:


import csv
import os
import sys
from pathlib import Path

import ccxt

# -----------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(""))))
sys.path.append(root + "/python")



# -----------------------------------------------------------------------------


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(
            exchange, max_retries, symbol, timeframe, fetch_since, limit
        )
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        print(
            len(all_ohlcv),
            symbol,
            "candles in total from",
            exchange.iso8601(all_ohlcv[0][0]),
            "to",
            exchange.iso8601(all_ohlcv[-1][0]),
        )
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return all_ohlcv


def write_to_csv(filename, exchange, data):
    p = Path("./data/raw/", str(exchange))
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)
    with Path(full_path).open("w+", newline="") as output_file:
        csv_writer = csv.writer(
            output_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerows(data)


def scrape_candles_to_csv(
    filename, exchange_id, max_retries, symbol, timeframe, since, limit
):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)(
        {"enableRateLimit": True,}  # required by the Manual
    )
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # save them to csv file
    write_to_csv(filename, exchange, ohlcv)
    print(
        "Saved",
        len(ohlcv),
        "candles from",
        exchange.iso8601(ohlcv[0][0]),
        "to",
        exchange.iso8601(ohlcv[-1][0]),
        "to",
        filename,
    )


# In[ ]:


scrape_candles_to_csv(
    "btc_usdt_1m.csv", "binance", 3, "BTC/USDT", "1m", "2019-01-0100:00:00Z", 1000
)
# scrape_candles_to_csv('./data/raw/binance/eth_btc_1m.csv', 'binance', 3, 'ETH/BTC', '1m', '2018-01-01T00:00:00Z', 1000)
# scrape_candles_to_csv('./data/raw/binance/ltc_btc_1m.csv', 'binance', 3, 'LTC/BTC', '1m', '2018-01-01T00:00:00Z', 1000)
# scrape_candles_to_csv('./data/raw/binance/xlm_btc_1m.csv', 'binance', 3, 'XLM/BTC', '1m', '2018-01-01T00:00:00Z', 1000)


# In[ ]:
