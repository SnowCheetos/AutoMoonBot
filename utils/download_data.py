import sqlite3
import yfinance as yf

def download_example(
        db_path:  str, 
        ticker:   str, 
        period:   str, 
        interval: str) -> None:
    
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
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

    data = yf.download(ticker, period=period, interval=interval)
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
        db_path="data/example.db", 
        ticker="SPY", 
        period="3mo", 
        interval="1h")