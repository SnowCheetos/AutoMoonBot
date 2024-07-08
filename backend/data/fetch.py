import asyncio
import aiohttp


async def fetch_data(session, symbol, month, size, interval, api_key):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}"
        f"&month={month}"
        f"&outputsize={size}"
        f"&interval={interval}"
        f"&apikey={api_key}"
    )
    async with session.get(url) as response:
        return await response.json()


async def fetch_news(session, start, end, api_key):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT"
        f"&time_from={start}"
        f"&time_to={end}"
        f"&limit=1000"
        f"&sort=EARLIEST"
        f"&apikey={api_key}"
    )
    async with session.get(url) as response:
        return await response.json()


async def fetch_prices(symbols, month, size, interval, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_data(session, symbol, month, size, interval, api_key)
            for symbol in symbols
        ]
        responses = await asyncio.gather(*tasks)
        return responses