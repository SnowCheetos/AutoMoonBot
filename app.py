import base64
import json
import asyncio
import aiofiles
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from backend.server import Server
from utils.tools import interval_map


with open("config.json", "r") as f:
    config = json.load(f)

logging.basicConfig(
    filename=config["log_file"],
    filemode=config["log_mode"],
    level=logging._nameToLevel[config["log_level"]])

logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="./static/"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/media", StaticFiles(directory="media"), name="media")

if config["live_data"] and "s" in config["interval"]:
    raise Exception("intervals less than 1 minute can only be used for back testing, not live data")

period, interval, db_path, bt_interval = None, None, None, config.get("backtest_interval")
if not config["live_data"]:
    with open("data/helper.json", "r") as f:
        data_helper = json.load(f).get(config["ticker"])
        if not data_helper:
            raise Exception("Ticker does not have data")
        period      = data_helper["period"]
        interval    = data_helper["interval"]
        db_path     = data_helper["file"]

server = Server(
    ticker=config["ticker"],
    period=period if period else config["period"],
    interval=interval if interval else config["interval"],
    queue_size=config["queue_size"],
    state_dim=config["state_dim"],
    action_dim=config["action_dim"],
    embedding_dim=config["embedding_dim"],
    inaction_cost=config["inaction_cost"],
    action_cost=config["action_cost"],
    device=config["device"],
    return_thresh=config["return_thresh"],
    feature_params=config["feature_params"],
    db_path=db_path if db_path else config["db_path"],
    live_data=config["live_data"],
    logger=logger,
    sharpe_cutoff=config["sharpe_cutoff"],
    inference_method=config["inference_method"],
    training_params=config["training_params"],
    retrain_freq=config["retrain_freq"],
    max_training_data=config["max_training_data"],
    alpha=config["alpha"],
    beta=config["beta"],
    gamma=config["gamma"],
)

async def model_data_update_loop(ws: WebSocket, s: Server):
    while ws.client_state != WebSocketState.DISCONNECTED:
        iv = config["interval"] if config["live_data"] else bt_interval
        await asyncio.sleep(interval_map[iv])
        action = s.consume_queue()
        if action:
            action["type"] = "action"
            await ws.send_json(data=action)

        data = s.tohlcv()
        data["type"] = "ohlc"
        await ws.send_json(data=data)

async def server_status_update_loop(ws: WebSocket, s: Server):
    while ws.client_state != WebSocketState.DISCONNECTED:
        if s.new_session:
            info = s.session_info
            await ws.send_json(data=info)
        data = s.status_report()
        await ws.send_json(data=data)
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    iv = config["interval"] if config["live_data"] else bt_interval
    server.start_timer(interval_map[iv])

@app.get("/")
async def home():
    return FileResponse("./static/index.html")

@app.websocket("/connect")
async def ws_handler(ws: WebSocket):
    model_loop, server_loop = None, None
    try:
        await ws.accept()
        model_loop  = asyncio.create_task(
            model_data_update_loop(ws, server))
        server_loop = asyncio.create_task(
            server_status_update_loop(ws, server))
        await asyncio.gather(model_loop, server_loop)

    except WebSocketDisconnect as e:
        logger.error(str(e))

    except Exception as e:
        logger.error(str(e))

    finally:
        for task in (model_loop, server_loop):
            if task: task.cancel()

@app.get("/tohlcv/last")
async def tohlcv_last():
    data = server.tohlcv()
    return JSONResponse(content=data)

@app.get("/tohlcv/all")
async def tohlcv_all():
    data = server.fetch_buffer()
    return JSONResponse(content=data)

@app.get("/session")
async def session_info():
    info = server.session_info
    info["live"]   = config["live_data"]
    info["record"] = config["record_frames"]
    return JSONResponse(content=info)

@app.post("/save_frame/{frame_id}")
async def save_frame(request: Request, frame_id: str):
    data = await request.json()
    frame = data["frame"]
    
    img_data = frame.replace("data:image/png;base64,", "")
    file_path = config["frames_dir"] + f"/{frame_id}.png"
    
    async with aiofiles.open(file_path, 'wb') as file:
        await file.write(base64.b64decode(img_data))
    
    return {"success": True, "message": "Image saved successfully", "file_path": file_path}

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

    if server.busy:
        return JSONResponse(status_code=400, content={"error": "there is another training thread running, wait for it to finish first"})

    server.train_model(
        episodes=int(header["episodes"]),
        learning_rate=float(header["learning_rate"]),
        momentum=float(header["momentum"]),
        max_grad_norm=float(header["max_grad_norm"]),
        portfolio_size=int(header["portfolio_size"])
    )
    return JSONResponse({"success": "training started, check server logs for more details"})

@app.get("/inference")
async def inference(request: Request):
    header = dict(request.headers)
    update = False
    if "update" in list(header.keys()):
        update = header["update"]

    action = server.run_inference(bool(update))
    if action is None:
        return JSONResponse(status_code=400, content={"error": "inference failed, check server logs for more details"})
    
    return JSONResponse({"action": action.name})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=29697)