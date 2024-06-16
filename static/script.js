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

var counter     = 0

var chart = new CanvasJS.Chart("chart-holder", {
    animationEnabled: true,
    theme: "light1", // "light1", "light2", "dark1", "dark2"
    exportEnabled: true,
    // title: {
    //     text: "Stonks"
    // },
    // subtitles: [{
    //     text: "Good"
    // }],
    axisX: {
        interval: 10,
        intervalType: "category",
        labelFontSize: 12,
        // valueFormatString: "MMM DD"
    },
    axisY: {
        prefix: "$",
        labelFontSize: 12,
        // title: "Price"
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
            // Handle action data
            if (data.action === "Buy") {
                appendAction(data.action, data.close, data.probability)
                startTrade(data.timestamp, data.close)
            } else if (data.action === "Sell") {
                appendAction(data.action, data.close, data.probability)
                finishTrade(data.close)
            }
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
            if (performanceRec.initPrice === 0) {
                performanceRec.initPrice = data.close
            } else {
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

initClientWebSocket()
// await fetchBuffer()

// async function fetchBuffer() {
//     try {
//         const response = await fetch("/tohlcv/all", {
//             method: "GET",
//             credentials: 'include',
//         });
        
//         const data = await response.json();
//         data.data.forEach(element => {
//             if (performanceRec.initPrice === 0) {
//                 performanceRec.initPrice = element.close
//             }
//             dataBuffer.push({
//                 x: counter, // new Date(element.timestamp * 1000),
//                 y: [
//                     element.open,
//                     element.high,
//                     element.low,
//                     element.close
//                 ]
//             });
//             counter ++

//             if (dataBuffer.length > dataBufferSize) {
//                 dataBuffer.shift();
//             }

//         });

//     } catch (err) {
//         console.error("Error initializing session:", err);
//     }

//     chart.render();
// }

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

    const percent = (performanceRec.totalGain-1) * 100
    item.innerHTML = `${performanceRec.totalGain > 1 ? '+' : ''}${(percent).toFixed(2)}%`
}

function updateBuyAndHold(close) {
    const item = document.getElementById('buy-and-hold')
    performanceRec.buyAndHold = close / performanceRec.initPrice
    
    const percent = (performanceRec.buyAndHold-1) * 100
    item.innerHTML = `${performanceRec.buyAndHold > 1 ? '+' : ''}${(percent).toFixed(2)}%`
}

function appendAction(action, close, probability) {
    const buffer = document.getElementById('actions-buffer')
    const listItem = document.createElement('li');

    var tag
    if (action === 'Buy') {
        tag = '<div class="buy-tag">'
    } else if (action === 'Sell') {
        tag = '<div class="sell-tag">'
    }

    listItem.innerHTML = `
    <div class="action-card">
        ${tag}${action}</div>
        <div>
            <div>${(close).toFixed(2)}</div>
            <div>${(probability*100).toFixed(2)}%</div>
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
    <div class="trade-card">
        ${tag}
        <div>
            <div>Bought at: ${(trade.entry).toFixed(2)}</div>
            <div>Sold at:   ${(trade.exit).toFixed(2)}</div>
        </div>
    </div>
    `

    buffer.insertBefore(listItem, buffer.firstChild)

    if (buffer.length > tradeBufferSize) {
        buffer.removeChild(buffer.lastChild)
    }
}