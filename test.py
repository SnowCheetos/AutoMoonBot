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
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "windows": [8, 12, 16, 20, 24, 30, 36, 42, 56, 64],
        },
        session_id='test',
        inf_interval=1,
        trn_interval=10,
        market_rep=['VTI', 'IWM', 'QQQ', 'EEM', 'VEA', 'IYR', 'VFH', 'BND', 'XLE', 'GLD']
    )

    session.start()