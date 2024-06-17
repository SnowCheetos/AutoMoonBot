const ws_protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

const actionBufferSize      = 20
const dataBufferSize        = 60
const tradeBufferSize       = 20

var client_ws          = null
var actionBuffer       = []
var dataBuffer         = []
var tradeBuffer        = {}
var currTradeTimeStamp = null
var performanceRec     = {
    initPrice:  0,
    totalGain:  1,
    buyAndHold: 1
}

var started     = false
var counter     = 0

initClientWebSocket()

var chart = new CanvasJS.Chart("chart-holder", {
    animationEnabled: true,
    theme: "light1", // "light1", "light2", "dark1", "dark2"
    exportEnabled: true,
    title: {
        text: "SPY",
        fontSize: 24
    },
    axisX: {
        interval: 10,
        intervalType: "category",
        labelFontSize: 12,
    },
    axisY: {
        prefix: "$",
        labelFontSize: 12,
    },
    toolTip: {
        content: "Index: {x}<br /><strong>Price:</strong><br />Open: {y[0]}, Close: {y[3]}<br />High: {y[1]}, Low: {y[2]}",
    },
    data: [{
        type: "candlestick",
        yValueFormatString: "$##0.00",
        dataPoints: dataBuffer,
        risingColor: "white",
        fallingColor: "black",
        color: "black"
    }]
});

function initClientWebSocket() {
    client_ws = new WebSocket(`${ws_protocol}//${window.location.host}/connect`);
    
    client_ws.onmessage = function(event) {
        const data = JSON.parse(event.data);    
        if (data.type === "action") {
            if (data.action === "Buy") {
                if (!started) {
                    started = true
                }
                appendAction(data.action, data.close, data.probability)
                startTrade(data.timestamp, data.close)
            } else if (data.action === "Sell") {
                appendAction(data.action, data.close, data.probability)
                finishTrade(data.close)
            }
        } else if (data.type === "report") {
            serverStatus(data)
        } else if (data.type === "ohlc") {
            dataBuffer.push({
                x: counter,
                y: [
                    data.open,
                    data.high,
                    data.low,
                    data.close
                ]
            });
            counter++;
            if (performanceRec.initPrice === 0 && started) {
                performanceRec.initPrice = data.close
            } else if (started) {
                updateBuyAndHold(data.close)
            }
            if (dataBuffer.length > dataBufferSize) {
                dataBuffer.shift()
            }

            chart.render()
        }
    };
    
    client_ws.onclose = function(event) {
        console.error('WebSocket closed:', event);
        setTimeout(initClientWebSocket, 1000);
    };

    client_ws.onerror = function(event) {
        console.error('WebSocket error:', event);
        client_ws.close();
    };
}

function serverStatus(status) {
    const modelStatus    = document.getElementById('model-status')
    const trainingStatus = document.getElementById('training-status')

    var modelStatusScript
    var trainingStatusScript
    if (status.ready) {
        modelStatusScript = `
        <div class="model-ready">
            Model Instance Ready
        </div>
        `
    } else {
        modelStatusScript = `
        <div class="model-not-ready">
            Model Unavailable
        </div>
        `
    }
    modelStatus.innerHTML = modelStatusScript

    if (status.training) {
        trainingStatusScript = `
        <div class="model-training">
            Training in Progress
        </div>
        `
    } else {
        trainingStatusScript = `
        <div class="model-not-training">
            Training State Idle
        </div>
        `
    }
    trainingStatus.innerHTML = trainingStatusScript
}

function startTrade(timestamp, close) {
    currTradeTimeStamp = timestamp
    tradeBuffer[timestamp] = {
        timestamp: timestamp,
        entry:     close,
        exit:      0.0
    }
}

function finishTrade(close) {
    tradeBuffer[currTradeTimeStamp].exit = close
    const trade = tradeBuffer[currTradeTimeStamp]
    appendTrade(trade)
    updateTotalGain(trade)
    currTradeTimeStamp = null
}

function updateTotalGain(trade) {
    const item = document.getElementById('total-gain')
    const outcome = trade.exit / trade.entry
    performanceRec.totalGain *= outcome

    const percent  = (performanceRec.totalGain-1) * 100
    item.innerHTML = `${performanceRec.totalGain > 1 ? '+' : ''}${(percent).toFixed(2)}%`
    item.color     = performanceRec.totalGain > 1 ? 'green' : 'red'
}

function updateBuyAndHold(close) {
    if (performanceRec.initPrice === 0) {
        return
    }
    const item = document.getElementById('buy-and-hold')
    performanceRec.buyAndHold = close / performanceRec.initPrice
    
    const percent  = (performanceRec.buyAndHold-1) * 100
    item.innerHTML = `${performanceRec.buyAndHold > 1 ? '+' : ''}${(percent).toFixed(2)}%`
    item.color     = performanceRec.buyAndHold > 1 ? 'green' : 'red'
}

function appendAction(action, close, probability) {
    const buffer = document.getElementById('actions-buffer')
    const listItem = document.createElement('li');

    var tag
    if (action === 'Buy') {
        tag = `<div class="buy-tag">${action}</div>`
    } else if (action === 'Sell') {
        tag = `<div class="sell-tag">${action}</div>`
    }

    listItem.innerHTML = `
    <div class="holder-with-ts">
        <div class="ts"> ${getCurrentTimestamp()} </div>
        <div class="action-card">
            ${tag}
            <div class="action-info">
                <div class="close-price">Price: $${(close).toFixed(2)}</div>
                <div class="certainty">Certainty: ${(probability*100).toFixed(2)}%</div>
            </div>
        </div>
    </div>
    `
    buffer.insertBefore(listItem, buffer.firstChild)

    if (buffer.length > actionBufferSize) {
        buffer.removeChild(buffer.lastChild)
    }
}

function appendTrade(trade) {
    const buffer = document.getElementById('logs')
    const listItem = document.createElement('li');
    const outcome = (trade.exit / trade.entry) - 1

    var tag
    if (outcome > 0) {
        tag = `
        <div class="win-tag">
            +${(outcome * 100).toFixed(2)}%
        </div>`
    } else {
        tag = `
        <div class="loss-tag">
            ${(outcome * 100).toFixed(2)}%
        </div>`
    }

    listItem.innerHTML = `
    <div class="holder-with-ts">
        <div class="ts"> ${getCurrentTimestamp()} </div>
        <div class="trade-card">
            ${tag}
            <div class="trade-info">
                <div>Bought at: $${(trade.entry).toFixed(2)}</div>
                <div>Sold at:   $${(trade.exit).toFixed(2)}</div>
            </div>
        </div>
    </div>
    `

    buffer.insertBefore(listItem, buffer.firstChild)

    if (buffer.length > tradeBufferSize) {
        buffer.removeChild(buffer.lastChild)
    }
}

function getCurrentTimestamp() {
    const now = new Date();
    
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0'); // Months are zero-based
    const day = String(now.getDate()).padStart(2, '0');
    
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    return `${year}/${month}/${day} ${hours}:${minutes}:${seconds}`;
}