const ws_protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

const actionBufferSize      = 20
const dataBufferSize        = 60
const tradeBufferSize       = 20

var chart              = null
var client_ws          = null
var dataBuffer         = []
var timeBuffers        = {}
var tradePoints        = []
var tradeBuffer        = {}
var currTradeTimeStamp = null
var performanceRec     = {
    initPrice:  0,
    totalGain:  1,
    buyAndHold: 1
}
var session = {
    live:     false,
    ticker:   null,
    period:   null,
    interval: null,
    record:   false
}

var started     = false
var counter     = 0

initSession()

function initChart() {
    chart = new CanvasJS.Chart("chart-container", {
        animationEnabled: true,
        theme: "light1", // "light1", "light2", "dark1", "dark2"
        exportEnabled: true,
        title: {
            text: `${session.live? 'Live: ': 'Back-Test: '} ${session.ticker} ${session.interval} ${session.live? '' : session.period}`,
            fontSize: 24
        },
        axisX: {
            labelFormatter: function() {
                return "";
            },
            intervalType: "number",
            stripLines: tradePoints
        },
        axisY: {
            prefix: "$",
            labelFontSize: 12,
        },
        toolTip: {
            contentFormatter: function (e) {
                var dataPoint = e.entries[0].dataPoint;
                var index = dataPoint.x;
                return "Time: " + timeBuffers[index] + "<br /><strong>Price:</strong><br />Open: " + dataPoint.y[0] + ", Close: " + dataPoint.y[3] + "<br />High: " + dataPoint.y[1] + ", Low: " + dataPoint.y[2];
            }
        },
        data: [
            {
                type: "candlestick",
                yValueFormatString: "$##0.00",
                dataPoints: dataBuffer,
                risingColor: "white",
                fallingColor: "black",
                color: "black"
            }
        ]
    })
}

function captureScreenshot() {
    var element = document.body

    html2canvas(element).then(function(canvas) {
        var imgData = canvas.toDataURL("image/png")

        fetch(`/save_frame/${counter}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ frame: imgData })
        }).then(response => response.json())
            .then(data => {
                console.log('Success:', data)
            }).catch((error) => {
                console.error('Error:', error)
            })
    })
}

function addVerticalDashedLine(index, color = "red", thickness = 2) {
    tradePoints.push({
        value: index,
        lineDashType: "dash",
        color: color,
        thickness: thickness
    })
    chart.render()
}

function addStripeLines(start, end, gain) {
    const existingStripLine = tradePoints.find(
        stripLine => stripLine.name === currTradeTimeStamp
    );
    const existingDoubledStripLine = tradePoints.find(
        stripLine => stripLine.name === currTradeTimeStamp + "_d"
    );

    const pctGain = (gain - 1) * 100;

    if (!existingStripLine) {
        tradePoints.push({
            name: currTradeTimeStamp,
            startValue: start,
            endValue: end,
            color: gain > 1 ? "#ECFFEE" : "#FFECEC",
            label: gain > 1 ? `+${pctGain.toFixed(2)}%` : `${pctGain.toFixed(2)}%`,
            labelAlign: "center",
            labelFontColor: "black",
            labelPlacement: "outside",
            labelBackgroundColor: "white",
            labelFontSize: 12
        })
    } else {
        const trade = tradeBuffer[currTradeTimeStamp]
        if (trade.doubled == 0) {
            existingStripLine.endValue = end;
            existingStripLine.color = gain > 1 ? "#ECFFEE" : "#FFECEC",
            existingStripLine.label = gain > 1 ? `+${pctGain.toFixed(2)}%` : `${pctGain.toFixed(2)}%`;
            existingStripLine.labelFontColor = gain > 1 ? "#6FFF5B" : "#FF5B5B";
        } else {
            if (!existingDoubledStripLine) {
                tradePoints.push({
                    name: currTradeTimeStamp + "_d",
                    startValue: trade.doubled,
                    endValue: end,
                    color: gain > 1 ? "#C5FFC4" : "#FFC4C4",
                    label: gain > 1 ? `+${pctGain.toFixed(2)}%` : `${pctGain.toFixed(2)}%`,
                    labelAlign: "center",
                    labelFontColor: "black",
                    labelPlacement: "outside",
                    labelBackgroundColor: "white",
                    labelFontSize: 12
                })
            } else {
                existingDoubledStripLine.endValue = end;
                existingDoubledStripLine.color = gain > 1 ? "#C5FFC4" : "#FFC4C4",
                existingDoubledStripLine.label = gain > 1 ? `+${pctGain.toFixed(2)}%` : `${pctGain.toFixed(2)}%`;
                existingDoubledStripLine.labelFontColor = gain > 1 ? "green" : "red";
            }
        }
    }

    chart.render()
}

function initSession() {
    fetch(`/session`)        
    .then(response => {
        if (response.status == 200) {
            response.json().then(data => {
                session.live     = data.live
                session.ticker   = data.ticker
                session.period   = data.period
                session.interval = data.interval
                session.record   = data.record

                initChart()
                initClientWebSocket()
            })
        } else if (response.status === 400) {
            console.error('400 received from server')
        }
    })
    .catch(error => 
        console.error('Error fetching device data:', error)
    )
}

function initClientWebSocket() {
    loadCache()
    client_ws = new WebSocket(`${ws_protocol}//${window.location.host}/connect`);
    
    client_ws.onmessage = function(event) {
        const data = JSON.parse(event.data);    
        data.data.forEach(element => {
            if (element.type === "new_session") {
                reset()
            } else if (element.type === "action") {
                if (element.action === "Buy") {
                    if (!started) {
                        started = true
                    }
                    appendAction(element.action, element.close, element.probability)
                    startTrade(element.timestamp, element.close, counter, element.amount)
                } else if (element.action === "Sell") {
                    appendAction(element.action, element.close, element.probability)
                    finishTrade(element.close, counter)
                } else if (element.action === "Double") {
                    appendAction(element.action, element.close, element.probability)
                    addVerticalDashedLine(counter, "purple", 1)
                    const trade = tradeBuffer[currTradeTimeStamp]
                    trade.doubled = counter
                    trade.entry   = 0.5 * (element.close + trade.entry)
                    trade.amount *= 2
                }
            } else if (element.type === "report") {
                serverStatus(element)
            } else if (element.type === "ohlc") {
                addToDataBuffer(element)
                if (performanceRec.initPrice === 0 && started) {
                    performanceRec.initPrice = element.close
                } else if (started) {
                    updateBuyAndHold(element.close)
                }
                
                if (currTradeTimeStamp) {
                    const trade = tradeBuffer[currTradeTimeStamp]
                    addStripeLines(trade.start, element.nounce, element.close / trade.entry)
                }
                
                if (data.type !== "report") {
                    saveCache()
                }
    
                if (session.record) {
                    captureScreenshot()
                }
                chart.render()
            }
        })
    }
    
    client_ws.onclose = function(event) {
        console.error('WebSocket closed:', event);
        setTimeout(initClientWebSocket, 1000);
    };

    client_ws.onerror = function(event) {
        console.error('WebSocket error:', event);
        client_ws.close();
    };
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000); // Convert timestamp to milliseconds

    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0'); // Months are zero-based
    const day = String(date.getDate()).padStart(2, '0');
    
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

function addToDataBuffer(data) {
    dataBuffer.push({
        x: counter,
        y: [
            data.open,
            data.high,
            data.low,
            data.close
        ]
    })

    timeBuffers[counter] = formatTimestamp(data.timestamp)

    counter = data.nounce

    if (dataBuffer.length > dataBufferSize) {
        dataBuffer.shift()
    }
}

function reset() {
    clearCache()
    client_ws          = null
    dataBuffer         = []
    tradeBuffer        = {}
    currTradeTimeStamp = null
    performanceRec     = {
        initPrice:  0,
        totalGain:  1,
        buyAndHold: 1
    }

    session     = {}
    timeBuffers = {}
    tradePoints = []

    started = false
    counter = 0
    window.location.reload()
}

function clearCache() {
    localStorage.clear()
}

function saveCache() {
    const cache = {
        dataBuffer:         dataBuffer,
        tradeBuffer:        tradeBuffer,
        currTradeTimeStamp: currTradeTimeStamp,
        performanceRec:     performanceRec,
        started:            started,
        counter:            counter,
        session:            session,
        timeBuffers:        timeBuffers,
        tradePoints:        tradePoints,
        actionsHTML:        document.getElementById('actions-buffer').innerHTML,
        tradesHTML:         document.getElementById('logs').innerHTML
    }
    localStorage.setItem('cache', JSON.stringify(cache));
}

function loadCache() {
    const data = localStorage.getItem('cache')
    if (!data) {
        return
    }
    const cache = JSON.parse(localStorage.getItem('cache'));

    dataBuffer         = cache.dataBuffer
    tradeBuffer        = cache.tradeBuffer
    currTradeTimeStamp = cache.currTradeTimeStamp
    performanceRec     = cache.performanceRec
    started            = cache.started
    counter            = cache.counter
    session            = cache.session
    tradePoints        = cache.tradePoints
    timeBuffers        = cache.timeBuffers

    initChart()

    if (performanceRec.totalGain !== 1) {
        const item       = document.getElementById('total-gain')
        const percent    = (performanceRec.totalGain-1) * 100
        item.innerHTML   = `${performanceRec.totalGain > 1 ? '+' : ''}${(percent).toFixed(2)}%`
        item.style.color = performanceRec.totalGain > 1 ? 'green' : 'red'
    }

    document.getElementById('actions-buffer').innerHTML = cache.actionsHTML
    document.getElementById('logs').innerHTML = cache.tradesHTML
}

function serverStatus(status) {
    if (status.done) {
        alert(`
            Back testing complete. 
            Click to return to home page.`)
    }

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
            Model Initializing
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

function startTrade(timestamp, close, idx, amount) {
    addVerticalDashedLine(idx, "green", 1)
    currTradeTimeStamp = timestamp
    tradeBuffer[timestamp] = {
        timestamp: timestamp,
        start:     counter,
        entry:     close,
        exit:      0.0,
        amount:    amount,
        doubled:   0
    }
}

function finishTrade(close, idx) {
    addVerticalDashedLine(idx, "red", 1)
    tradeBuffer[currTradeTimeStamp].exit = close
    tradeBuffer[currTradeTimeStamp].end = idx
    const trade = tradeBuffer[currTradeTimeStamp]
    appendTrade(trade)
    addStripeLines(trade.start, trade.end, (trade.exit / trade.entry - 1) * trade.amount + 1)
    updateTotalGain(trade, true)
    currTradeTimeStamp = null
}

function updateTotalGain(trade) {
    const item = document.getElementById('total-gain')
    const outcome = (trade.exit / trade.entry - 1) * trade.amount + 1
    performanceRec.totalGain *= outcome

    const percent    = (performanceRec.totalGain-1) * 100
    item.innerHTML   = `${performanceRec.totalGain > 1 ? '+' : ''}${(percent).toFixed(2)}%`
    item.style.color = performanceRec.totalGain > 1 ? 'green' : 'red'
}

function updateBuyAndHold(close) {
    if (performanceRec.initPrice === 0) {
        return
    }
    const item = document.getElementById('buy-and-hold')
    performanceRec.buyAndHold = close / performanceRec.initPrice
    
    const percent    = (performanceRec.buyAndHold-1) * 100
    item.innerHTML   = `${performanceRec.buyAndHold > 1 ? '+' : ''}${(percent).toFixed(2)}%`
    item.style.color = performanceRec.buyAndHold > 1 ? 'green' : 'red'
}

function appendAction(action, close, probability) {
    const buffer = document.getElementById('actions-buffer')
    const listItem = document.createElement('li');

    var tag
    if (action === 'Buy') {
        tag = `<div class="buy-tag">${action}</div>`
    } else if (action === 'Sell') {
        tag = `<div class="sell-tag">${action}</div>`
    } else if (action === 'Double') {
        tag = `<div class="double-tag">${action}</div>`
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
    listItem.classList.add('fade-in');
    buffer.insertBefore(listItem, buffer.firstChild)

    if (buffer.length > actionBufferSize) {
        buffer.removeChild(buffer.lastChild)
    }
    listItem.offsetHeight; 
    listItem.classList.add('fade-in-visible');
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
                <div>Avg cost: $${(trade.entry).toFixed(2)}</div>
                <div>Sold at:   $${(trade.exit).toFixed(2)}</div>
            </div>
        </div>
    </div>
    `
    listItem.classList.add('fade-in');
    buffer.insertBefore(listItem, buffer.firstChild)

    if (buffer.length > tradeBufferSize) {
        buffer.removeChild(buffer.lastChild)
    }
    listItem.offsetHeight; 
    listItem.classList.add('fade-in-visible');
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