import sqlite3
import pandas as pd

if __name__ == '__main__':
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect("data/database.sqlite")
    df = pd.read_sql_query("SELECT * from Country", con)
    # Verify that result of SQL query is stored in the dataframe
    print(df.head())

    con.close()
