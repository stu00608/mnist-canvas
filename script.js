const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.1;
const INFERENCE_SIZE = 28;

let options = { willReadFrequently: true };
const canvas = document.getElementById("canvas");
const hiddenCanvas = document.getElementById("hiddenCanvas");
const loading = document.getElementById("loading");
const ctx = canvas.getContext("2d", options);
const hiddenCanvasCtx = hiddenCanvas.getContext("2d", options);
const rect = canvas.getBoundingClientRect();
hiddenCanvasCtx.scale(CANVAS_SCALE, CANVAS_SCALE);

ctx.lineWidth = 15;
ctx.lineCap = 'round'
ctx.lineJoin = "round";
ctx.strokeStyle = "#000000"

const hasTouchEvent = 'ontouchstart' in window ? true : false;

let isMouseActive = false;
let x1 = 0;
let y1 = 0;
let x2 = 0;
let y2 = 0;

const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("onnx_model.onnx");

async function updatePredictions() {
    // Get the predictions for the canvas data.

    hiddenCanvasCtx.drawImage(canvas, 0, 0);
    const hiddenImgData = hiddenCanvasCtx.getImageData(0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
    var data = hiddenImgData.data;
    var gray_data = [];

    for (var i = 3; i < data.length; i += 4) {
        pix = data[i] / 255;
        pix = (pix - 0.1307) / 0.3081
        gray_data.push(pix);
    }

    const input = new onnx.Tensor(new Float32Array(gray_data), "float32", [1, 1, INFERENCE_SIZE, INFERENCE_SIZE]);

    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = softmax(outputTensor.data);
    const maxPrediction = Math.max(...predictions);
    const predictLabel = predictions.findIndex((n) => n == maxPrediction);

    for (let i = 0; i < predictions.length; i++) {
        const bar = document.getElementById(`bar-${i}`);
        const num = document.getElementById(`num-${i}`);
        bar.style.width = `${predictions[i] * 200}px`;
        if (predictLabel == i) {
            bar.style.backgroundColor = "#6F38C5";
            num.style.fontWeight = "bold";
        }
        else {
            bar.style.backgroundColor = "#87A2FB";
            num.style.fontWeight = "";
        }
    }

    // console.log("------------------------")
    // for (let i = 0; i < predictions.length; i++) {
    //     console.log(`${i}: ${predictions[i]}`)
    // }
    // console.log(`Prediction = ${predictions.indexOf(maxPrediction)}`)
    // console.log("------------------------")
    // ctx.scale(1.0, 1.0);
}

function softmax(arr) {
    return arr.map(function (value, index) {
        return Math.exp(value) / arr.map(function (y /*value*/) { return Math.exp(y) }).reduce(function (a, b) { return a + b })
    })
}

function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    hiddenCanvasCtx.setTransform(CANVAS_SCALE, 0, 0, CANVAS_SCALE, 0, 0);
    hiddenCanvasCtx.clearRect(0, 0, hiddenCanvasCtx.canvas.width / CANVAS_SCALE, hiddenCanvasCtx.canvas.height / CANVAS_SCALE);
}

function clearBar() {
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`bar-${i}`);
        bar.style.width = "0px";
    }
}

function clearNumHighlight() {
    for (let i = 0; i < 10; i++) {
        const num = document.getElementById(`num-${i}`);
        num.style.color = "#000000";
        num.style.fontWeight = "";
    }
}

function getPos(x, y) {
    return {
        x: Math.round((x - rect.left) / (rect.right - rect.left) * canvas.width),
        y: Math.round((y - rect.top) / (rect.bottom - rect.top) * canvas.height)
    }
}

// Prevent scrolling when touching the canvas
function touchStart(e) {
    if (e.target == canvas) {
        e.preventDefault();
        isMouseActive = true;
        if (hasTouchEvent) {
            var pos = getPos(e.touches[0].clientX, e.touches[0].clientY);
        }
        else {
            var pos = getPos(e.clientX, e.clientY);
        }
        x1 = pos.x;
        y1 = pos.y;
    }
}

function touchEnd(e) {
    if (e.target == canvas) {
        isMouseActive = false;
    }
}

function touchMove(e) {
    if (e.target == canvas) {
        e.preventDefault();

        if (!isMouseActive) {
            return
        }
        if (hasTouchEvent) {
            var pos = getPos(e.touches[0].clientX, e.touches[0].clientY);
        }
        else {
            var pos = getPos(e.clientX, e.clientY);
        }
        x2 = pos.x;
        y2 = pos.y;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        x1 = x2;
        y1 = y2;

        updatePredictions();
    }
}

document.body.addEventListener("mouseout", function (e) {
    if (!e.relatedTarget || e.relatedTarget.nodeName === "HTML") {
        isMouseDown = false;
    }
});

loadingModelPromise.then(() => {
    console.log("Successfully loaded model.");

    $(".lds-ring").hide();

    $("#clear").click(() => {
        clearArea();
        clearBar();
        clearNumHighlight();
    });

    if (hasTouchEvent) {
        document.body.addEventListener("touchstart", touchStart, { passive: false });
        document.body.addEventListener("touchmove", touchMove, { passive: false });
        document.body.addEventListener("touchend", touchEnd, { passive: false });
    }
    else {
        canvas.addEventListener("mousedown", touchStart);
        canvas.addEventListener("mousemove", touchMove);
        canvas.addEventListener("mouseup", touchEnd);
    }

})
