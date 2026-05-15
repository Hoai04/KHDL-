import pandas as pd
from sqlalchemy import create_engine

print("Đang đọc file...")

df = pd.read_csv("processed_online_shoppers_fixed.csv")

print(df.head())

engine = create_engine(
    "mssql+pyodbc://localhost/ONLINE_SHOPPERS?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

df.to_sql(
    "processed_online_shoppers",
    engine,
    if_exists="replace",
    index=False
)

print("Import thành công!")