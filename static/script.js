const ws_protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'

var client_ws
var dataPoints  = []
var buyIndices  = []
var sellIndices = []
var maxPoints   = 40
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
        dataPoints: dataPoints,
        risingColor: "white",
        fallingColor: "black",
        color: "black"
    }]
});

function initClientWebSocket() {
    client_ws = new WebSocket(`${ws_protocol}//${window.location.host}/connect`);
    
    client_ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log(data);
        
        if (data.type === "action") {
            // Handle action data
            if (data.action === 0) {
                buyIndices.push(counter)
            } else if (data.action === 2) {
                sellIndices.push(counter)
            }
        } else if (data.type === "ohlc") {
            // Handle OHLC price data
            dataPoints.push({
                x: counter, // or use new Date(data.timestamp * 1000) if you need actual dates
                y: [
                    data.open,
                    data.high,
                    data.low,
                    data.close
                ]
            });
            counter++;

            if (dataPoints.length > maxPoints) {
                dataPoints.shift()
            }

            updateChart(chart)
            chart.render()
        }
    };
    
    client_ws.onclose = function(event) {
        console.error('WebSocket closed:', event);
        // Optionally attempt to reconnect after a delay
        setTimeout(initClientWebSocket, 1000);
    };

    client_ws.onerror = function(event) {
        console.error('WebSocket error:', event);
        // Optionally close the WebSocket to trigger the onclose event
        client_ws.close();
    };
}

initClientWebSocket()
await fetchBuffer()

async function fetchBuffer() {
    try {
        const response = await fetch("/tohlcv/all", {
            method: "GET",
            credentials: 'include',
        });
        
        const data = await response.json();
        data.data.forEach(element => {
            dataPoints.push({
                x: counter, // new Date(element.timestamp * 1000),
                y: [
                    element.open,
                    element.high,
                    element.low,
                    element.close
                ]
            });
            counter ++

            if (dataPoints.length > maxPoints) {
                dataPoints.shift();
            }

        });

    } catch (err) {
        console.error("Error initializing session:", err);
    }

    chart.render();
}

function updateChart(chart) {
    for(var i=0; i<chart.options.data[0].dataPoints.length; i++) {
        var dataPoint = chart.options.data[0].dataPoints[i];
        if (i in buyIndices) {
            dataPoint.indexLabel = "buy";
            dataPoint.color = "green";
        } else if (i in sellIndices) {
            dataPoint.indexLabel = "sell";
            dataPoint.color = "red";
        }
    }
}