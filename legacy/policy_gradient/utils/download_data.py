import sqlite3
import yfinance as yf

def download_example(
        db_path:  str, 
        ticker:   str, 
        period:   str, 
        interval: str,
        flush:    bool=True) -> None:
    
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    if flush:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.commit()

    query = """
    CREATE TABLE IF NOT EXISTS data (
        id         INTEGER PRIMARY KEY,
        timestamp  REAL,
        open       REAL,
        high       REAL,
        low        REAL,
        close      REAL,
        volume     REAL
    )
    """
    cursor.execute(query)

    data = yf.download(ticker, period=period, interval=interval).dropna()
    for idx in data.index:
        ts = idx.timestamp()
        row = data.loc[idx]
        cursor.execute("""
        INSERT INTO data (timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (ts, row.Open, row.High, row.Low, row.Close, row.Volume))

    con.commit()
    con.close()


if __name__ == "__main__":
    download_example(
        db_path  = "data/examples/JPY.db", 
        ticker   = "JPY=X", 
        period   = "2y",
        interval = "1h",
        flush    = True)