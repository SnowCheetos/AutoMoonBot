import pytest
from backend.session import Session

@pytest.fixture
def session():
    return Session(
        ticker='SPY',
        interval='1h',
        buffer_size=100,
        device='cpu',
        feature_config={
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "windows": [8, 10, 12, 14, 16, 18, 20],
        },
        session_id='test',
        live=False,
        db_path='../data',
        market_rep=['QQQ', 'USO', 'GLD']
    )

def test_graph_building(session: Session):
    f, c = session._loader.features
    data = session._build_graph(f, c)

    assert data.x.size(0) == 4, 'graph returned the wrong sized node features'