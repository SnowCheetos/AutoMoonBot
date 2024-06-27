import pytest
from backend.session import Session

TICKER = 'SPY'
INTERV = '1h'
BUFFER = 10
DEVICE = 'cpu'
PRELOA = True
CONFIG = {
    "columns": ["Open", "High", "Low", "Close", "Volume"],
    "windows": [8, 16, 32],
}
SESSID = 'test'
LIVEDT = False
DBPATH = '../data'
MARKET = ['QQQ', 'USO', 'GLD']

@pytest.fixture
def session():
    return Session(
        ticker         = TICKER,
        interval       = INTERV,
        buffer_size    = BUFFER,
        device         = DEVICE,
        preload        = PRELOA,
        feature_config = CONFIG,
        session_id     = SESSID,
        live           = LIVEDT,
        db_path        = DBPATH,
        market_rep     = MARKET)

def test_graph_building(session: Session):
    f, c = session._loader.features
    data = session._build_graph(f, c)

    assert data['graph'].x.size(0) == 4, 'graph returned the wrong sized node features'

def test_fetch_next(session: Session):
    data = session._fetch_next()

    assert data['graph'].x.size(0) == 4, 'graph returned the wrong sized node features'

def test_fill_dataset(session: Session):
    session._fill_dataset()

    dataset = session.dataset
    assert len(dataset) == BUFFER, 'dataset did not get filled properly'