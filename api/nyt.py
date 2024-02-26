from pynytimes import NYTAPI
import pandas as pd

nyt = NYTAPI("bBgxuC4ObtwuA5GSnAKyYqH63UDIAGD3", parse_dates=True)

articles = nyt.article_search(results=1000)

df = pd.DataFrame.from_records(articles)

csv_file = "api/articles.csv"

df.to_csv(csv_file, index=False)