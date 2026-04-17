let session;

let currentImageData = null; 
async function loadModel() {
    session = await ort.InferenceSession.create("fruit_cnn.onnx");
}

async function predict(imageData) {
    const tensor = new ort.Tensor("float32", imageData, [1, 3, 64, 64]);

    const results = await session.run({
        input: tensor
    });
    return results.output.data;
}

const classes = ["apple", "banana", "orange"];

function getPrediction(output) {
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
        if (output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }
    return classes[maxIndex];
}

function getPrediction(output) {
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
        if (output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }
    return classes[maxIndex];
}

async function handleImage(file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = () => {
        document.getElementById("preview").src = img.src;

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        canvas.width = 64;
        canvas.height = 64;
        ctx.drawImage(img, 0, 0, 64, 64);

        const imageData = ctx.getImageData(0, 0, 64, 64).data;

        let input = new Float32Array(1 * 3 * 64 * 64);

        let pixelIndex = 0;
        for (let i = 0; i < imageData.length; i += 4) {
            const r = imageData[i] / 255;
            const g = imageData[i + 1] / 255;
            const b = imageData[i + 2] / 255;

            input[pixelIndex] = r; // R
            input[pixelIndex + 64 * 64] = g; // G
            input[pixelIndex + 2 * 64 * 64] = b; // B

            pixelIndex++;
        }

        currentImageData = input;

        document.getElementById("result").innerText = "Image ready. Click Calculate.";
    };
}

document.getElementById("upload").addEventListener("change", (e) => {
    handleImage(e.target.files[0]);
});

const dropArea = document.getElementById("drop-area");

["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, e => e.preventDefault());
});

dropArea.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    handleImage(file);
});

document.getElementById("predictBtn").addEventListener("click", async () => {
    if (!currentImageData) {
        document.getElementById("result").innerText = "Upload image first.";
        return;
    }

    const output = await predict(currentImageData);
    const label = getPrediction(output);

    document.getElementById("result").innerText = "Prediction: " + label;
});
console.log(output);

loadModel();