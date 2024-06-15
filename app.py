import json
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from backend.server import Server


with open("config.json", "r") as f:
    config = json.load(f)

logging.basicConfig(
    filename=config["log_file"],
    filemode=config["log_mode"],
    level=logging._nameToLevel[config["log_level"]])

logger = logging.getLogger(__name__)

app = FastAPI()

server = Server(
    ticker=config["ticker"],
    period=config["period"],
    interval=config["interval"],
    queue_size=config["queue_size"],
    state_dim=config["state_dim"],
    action_dim=config["action_dim"],
    embedding_dim=config["embedding_dim"],
    inaction_cost=config["inaction_cost"],
    action_cost=config["action_cost"],
    device=config["device"],
    return_thresh=config["return_thresh"],
    feature_params=config["feature_params"],
    db_path=config["db_path"],
    logger=logger
)

@app.get("/tohlcv")
async def tohlcv():
    data = server.tohlcv()
    return JSONResponse(content=json.dumps(data))

@app.get("/train")
async def train(request: Request):
    header = dict(request.headers)
    keys = list(header.keys())

    required_keys = [
        "episodes",
        "learning_rate",
        "momentum",
        "max_grad_norm",
        "portfolio_size"]
    
    for key in required_keys:
        if key not in keys:
            return JSONResponse(status_code=400, content={"error": f"{key} not found in header"})

    server.train_model(
        episodes=int(header["episodes"]),
        learning_rate=float(header["learning_rate"]),
        momentum=float(header["momentum"]),
        max_grad_norm=float(header["max_grad_norm"]),
        portfolio_size=int(header["portfolio_size"])
    )
    return JSONResponse({"success": "training started"})

@app.get("/inference")
async def inference(request: Request):
    header = dict(request.headers)
    update = True
    if "update" in list(header.keys()):
        update = header["update"]

    action = server.run_inference(bool(update))
    if action is None:
        return JSONResponse(status_code=400, content={"error": "inference failed, check server logs for more details"})
    
    return JSONResponse({"action": action.name})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=29697)