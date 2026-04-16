let session;

async function loadModel() {
    session = await ort.InferenceSession.create("fruit_model.onnx");
}

async function predict(imageData) {
    const tensor = new ort.Tensor("float32", imageData, [1, 3, 64, 64]);

    const feeds = { input: tensor };
    const results = await session.run(feeds);

    return results.output.data;
}

document.getElementById("upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        canvas.width = 64;
        canvas.height = 64;
        ctx.drawImage(img, 0, 0, 64, 64);

        const imageData = ctx.getImageData(0, 0, 64, 64).data;

        let input = [];
        for (let i = 0; i < imageData.length; i += 4) {
            input.push(imageData[i] / 255);     // R
            input.push(imageData[i+1] / 255);   // G
            input.push(imageData[i+2] / 255);   // B
        }

        const output = await predict(new Float32Array(input));
        document.getElementById("result").innerText = "Prediction: " + output;
    };
});

loadModel();
