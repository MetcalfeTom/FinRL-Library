import datetime
import json

import pandas as pd
from pynytimes import NYTAPI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader

df = YahooDownloader(
    start_date="2009-01-01", end_date="2021-01-01", ticker_list=config.DOW_30_TICKER
).fetch_data()

nyt = NYTAPI("KEY")
encoder = SentenceTransformer("msmarco-distilroberta-base-v2")


results = {}
# results = pd.read_csv("encodings.csv").to_dict("list")
current_date = datetime.datetime(year=2000, month=6, day=1)

for datestring in tqdm(df.date.unique()):
    if isinstance(datestring, str):
        date = datetime.datetime.strptime(datestring, "%Y-%m-%d")
        if current_date.month != date.month:
            fname = f"{datestring}.json"
            data = nyt.archive_metadata(date=date)
            with open(f"nyt/{fname}", "w+") as f:
                json.dump(data, f, indent=2)
            current_date = date

        sentences = []
        for doc in data:
            pub_date = doc.get("pub_date")
            pub_date = datetime.datetime.strptime(pub_date[:10], "%Y-%m-%d")

            if pub_date == date:
                sentences.append(doc.get("abstract"))
            elif pub_date > date:
                break

        embedding = encoder.encode(".  ".join(sentences), show_progress_bar=False)
        row = {datestring: embedding.tolist()}
        results.update(row)

        encodings = pd.DataFrame(results)
        encodings.to_csv("encodings.csv", index=False)
