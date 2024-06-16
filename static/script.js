
var dataPoints = [];
var counter = 0

var chart = new CanvasJS.Chart("chart-holder", {
    animationEnabled: true,
    theme: "light1", // "light1", "light2", "dark1", "dark2"
    exportEnabled: true,
    title: {
        text: "Stonks"
    },
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
        position: "nearest",
        animationEnabled: true,
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
        });

    } catch (err) {
        console.error("Error initializing session:", err);
    }

    chart.render();
}

async function fetchLatest() {
    try {
        const response = await fetch("/tohlcv/last", {
            method: "GET",
            credentials: 'include',
        });
        
        const data = await response.json();
        
        dataPoints.push({
            x: counter, // new Date(element.timestamp * 1000),
            y: [
                data.open,
                data.high,
                data.low,
                data.close
            ]
        });
        counter ++

    } catch (err) {
        console.error("Error initializing session:", err);
    }

    chart.render();
}