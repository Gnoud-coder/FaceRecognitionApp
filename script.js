const container = document.querySelector('#container');
const fileInput = document.querySelector('#file-input');

async function loadTrainingData() {
    const labels = ['Hoang Duong'];

    const faceDescriptors = [];
    for (const label of labels) {
        const descriptors = [];
        for (let i = 1; i <= 1; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpeg`);
            const detection = await faceapi
                .detectSingleFace(image)
                .withFaceLandmarks()
                .withFaceDescriptor();
            descriptors.push(detection.descriptor);
            console.log(i);
        }
        faceDescriptors.push(
            new faceapi.LabeledFaceDescriptors(label, descriptors)
        );
        Toastify({
            text: `Trained data of ${label}!`,
        }).showToast();
    }

    return faceDescriptors;
}

let faceMatcher;
async function init() {
    await Promise.all([
        await faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        await faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    ]);

    Toastify({ text: 'Download detection model successfully' }).showToast();

    const trainingData = await loadTrainingData();
    faceMatcher = new faceapi.FaceMatcher(trainingData, 0.45);
}
init();

fileInput.addEventListener('change', async (e) => {
    const file = fileInput.files[0];

    const img = await faceapi.bufferToImage(file);
    const canvas = faceapi.createCanvasFromMedia(img);

    container.innerHTML = '';
    container.append(img);
    container.append(canvas);

    const size = { width: img.width, height: img.height };
    faceapi.matchDimensions(canvas, size);

    const detections = await faceapi
        .detectAllFaces(img)
        .withFaceLandmarks()
        .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, size);

    // faceapi.draw.drawDetections(canvas, resizedDetections);
    for (detection of resizedDetections) {
        const box = detection.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
            label: faceMatcher.findBestMatch(detection.descriptor).toString(),
        });
        drawBox.draw(canvas);
    }
});
