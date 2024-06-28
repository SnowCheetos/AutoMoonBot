import logging
from backend.session import Session

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    session = Session(
        ticker='SPY',
        interval='1h',
        buffer_size=-1,
        device='cpu',
        preload=True,
        combile_models=False,
        feature_config={
            "tickers": ['QQQ', 'USO', 'GLD'],
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "windows": [8, 16, 32],
        },
        session_id='test',
        inf_interval=1,
        trn_interval=10,
        market_rep=['QQQ', 'USO', 'GLD']
    )

    session.start()