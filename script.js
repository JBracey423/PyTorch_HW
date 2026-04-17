const imageUpload = document.getElementById('imageUpload');
const predictButton = document.getElementById('predictButton');
const imagePreview = document.getElementById('image-preview');
const predictionResult = document.getElementById('prediction-result');
const loadingIndicator = document.getElementById('loading');

let session;
let imageTensor; 
const MODEL_PATH = './fruit_cnn.onnx';
const IMAGE_SIZE = 64;
const LABELS = ['apple', 'banana', 'non-fruit', 'orange'];


function normalize(tensor) {
    const mean = [0.5, 0.5, 0.5];
    const std = [0.5, 0.5, 0.5];
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            tensor[c * IMAGE_SIZE * IMAGE_SIZE + i] = (tensor[c * IMAGE_SIZE * IMAGE_SIZE + i] - mean[c]) / std[c];
        }
    }
    return tensor;
}

async function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    ctx.drawImage(imageElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
    const input = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);

    let offset = 0;
    for (let i = 0; i < imageData.length; i += 4) {
        // R G B
        input[offset + 0 * IMAGE_SIZE * IMAGE_SIZE] = imageData[i] / 255.0; // R
        input[offset + 1 * IMAGE_SIZE * IMAGE_SIZE] = imageData[i + 1] / 255.0; // G
        input[offset + 2 * IMAGE_SIZE * IMAGE_SIZE] = imageData[i + 2] / 255.0; // B
        offset++;
    }
    
    const normalizedInput = normalize(input);

    imageTensor = new ort.Tensor("float32", normalizedInput, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

async function loadModel() {
    loadingIndicator.style.display = 'block';
    try {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('ONNX model loaded successfully!');
        loadingIndicator.style.display = 'none';
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
        predictionResult.innerText = 'Error loading model.';
        loadingIndicator.style.display = 'none';
    }
}

async function runInference() {
    if (!session || !imageTensor) {
        predictionResult.innerText = 'Model not loaded or image not processed.';
        return;
    }

    predictionResult.innerText = 'Predicting...';
    try {
        const feeds = {'input': imageTensor}; 
        const results = await session.run(feeds);
        const output = results[Object.keys(results)[0]].data; 

        let maxProb = -1;
        let predictedClassIndex = -1;
        for (let i = 0; i < output.length; i++) {
            if (output[i] > maxProb) {
                maxProb = output[i];
                predictedClassIndex = i;
            }
        }

        const predictedLabel = LABELS[predictedClassIndex];
        predictionResult.innerText = `Prediction: ${predictedLabel}`;
    } catch (e) {
        console.error('Failed to run inference:', e);
        predictionResult.innerText = 'Error during prediction.';
    }
}

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            predictButton.disabled = false;
            predictionResult.innerText = ''; // Clear previous results
        };
        reader.readAsDataURL(file);
    }
});

imagePreview.addEventListener('load', async () => {
    await preprocessImage(imagePreview);
});

predictButton.addEventListener('click', runInference);

loadModel();