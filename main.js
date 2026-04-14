// Initialize variables
let featureExtractor;
let classifier;
let video;
let isVideoReady = false;
let isPredicting = false;

// State management
let classesData = {};
let classCounter = 0;
const MIN_SAMPLES_PER_CLASS = 5;
const MIN_CLASSES = 2;

// DOM Elements
const classesContainer = document.getElementById('classes-container');
const addClassBtn = document.getElementById('add-class-btn');
const trainBtn = document.getElementById('train-btn');
const exportBtn = document.getElementById('export-btn');
const predictBtn = document.getElementById('predict-btn');
const webcamVideo = document.getElementById('webcam-video');
const trainingStatus = document.getElementById('training-status');
const trainingProgress = document.getElementById('training-progress');
const predictionResults = document.getElementById('prediction-results');

const epochsInput = document.getElementById('epochs-input');
const batchInput = document.getElementById('batch-input');
const lrInput = document.getElementById('lr-input');

// Initialize ml5 FeatureExtractor
async function init() {
    try {
        // Obter acesso à webcam
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = stream;
        
        webcamVideo.onloadedmetadata = () => {
            isVideoReady = true;
        };

        // Initialize MobileNet Feature Extractor
        featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded);
        
    } catch (err) {
        console.error("Erro ao acessar webcam", err);
        alert("Erro ao acessar a webcam. Você ainda pode usar o upload de imagens.");
        // Fallback to purely upload mode
        featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded);
    }
}

function modelLoaded() {
    console.log('MobileNet carregado!');
    classifier = featureExtractor.classification(webcamVideo, videoReady);
    // Add first two classes by default
    addClass('Classe 1');
    addClass('Classe 2');
}

function videoReady() {
    console.log('Video pronto para o classificador!');
}

// UI and Class Management
function addClass(defaultName = '') {
    const id = `class-${classCounter++}`;
    const name = defaultName || `Nova Classe`;
    
    classesData[id] = {
        name: name,
        samples: 0,
        images: []
    };

    const template = document.getElementById('class-template');
    const clone = template.content.cloneNode(true);
    const card = clone.querySelector('.class-card');
    card.dataset.id = id;

    const nameInput = clone.querySelector('.class-name');
    nameInput.value = name;
    nameInput.addEventListener('change', (e) => {
        classesData[id].name = e.target.value;
    });

    clone.querySelector('.delete-class-btn').addEventListener('click', () => removeClass(id, card));
    
    // Webcam Capture
    let captureInterval;
    const captureBtn = clone.querySelector('.capture-btn');
    captureBtn.addEventListener('mousedown', () => {
        if (!isVideoReady) return alert("Webcam não está pronta.");
        captureInterval = setInterval(() => addSample(id, webcamVideo), 100);
    });
    captureBtn.addEventListener('mouseup', () => clearInterval(captureInterval));
    captureBtn.addEventListener('mouseleave', () => clearInterval(captureInterval));

    // Image Upload
    const uploadInput = clone.querySelector('input[type="file"]');
    uploadInput.addEventListener('change', (e) => handleUpload(e, id));

    classesContainer.appendChild(clone);
    updateTrainButtonState();
}

function removeClass(id, cardElement) {
    if (Object.keys(classesData).length <= 2) {
        return alert("Você precisa de pelo menos 2 classes.");
    }
    delete classesData[id];
    cardElement.remove();
    updateTrainButtonState();
}

function handleUpload(event, classId) {
    const files = event.target.files;
    if (!files) return;

    for(let i=0; i<files.length; i++) {
        const file = files[i];
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                addSample(classId, img, img.src);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

function addSample(classId, imageSource, previewUrl = null) {
    if (!classifier) return;

    // Adiciona ao modelo do ml5
    classifier.addImage(imageSource, classesData[classId].name);
    
    classesData[classId].samples++;
    
    // Atualiza a UI
    const card = document.querySelector(`.class-card[data-id="${classId}"]`);
    if(card) {
        card.querySelector('.sample-count span').innerText = classesData[classId].samples;
        
        // Adiciona miniatura (apenas mantemos algumas p/ não travar memória)
        const previewContainer = card.querySelector('.samples-preview');
        if (previewContainer.children.length < 15) {
            const img = document.createElement('img');
            img.src = previewUrl || captureVideoFrame(imageSource);
            previewContainer.appendChild(img);
        }
    }

    updateTrainButtonState();
}

function captureVideoFrame(videoEl) {
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.5);
}

function updateTrainButtonState() {
    const classIds = Object.keys(classesData);
    let validClasses = 0;

    classIds.forEach(id => {
        if (classesData[id].samples >= MIN_SAMPLES_PER_CLASS) {
            validClasses++;
        }
    });

    if (classIds.length >= MIN_CLASSES && validClasses >= MIN_CLASSES) {
        trainBtn.disabled = false;
        trainBtn.innerText = "Treinar Modelo";
    } else {
        trainBtn.disabled = true;
        trainBtn.innerText = `Requer min. ${MIN_SAMPLES_PER_CLASS} amostras/classe`;
    }
}

// Training Loop
trainBtn.addEventListener('click', () => {
    trainBtn.disabled = true;
    trainingStatus.innerText = "Preparando treinamento...";
    trainingProgress.style.width = "5%";

    // Set advanced hyperparameters
    classifier.epochs = Number(epochsInput.value) || 50;
    classifier.batchSize = Number(batchInput.value) || 0.25;
    
    // Atualização: ml5 featureExtractor training callback passa a perda (loss) por epoch
    classifier.train((lossValue) => {
        if (lossValue) {
            trainingStatus.innerText = `Treinando... Loss: ${lossValue.toFixed(4)}`;
            // Uma aproximação visual, featureExtractor não manda % exato nativamente de forma exposta na callback simples no ml5 antigo, simulamos incremento visual ou calculamos base nos epochs se possível
            // Como é contínuo, vamos apenas fazer um efeito de progresso
            let currentWidth = parseFloat(trainingProgress.style.width) || 5;
            trainingProgress.style.width = Math.min(currentWidth + 2, 95) + "%";
        } else {
            // Concluído
            trainingStatus.innerText = "Treinamento Concluído!";
            trainingProgress.style.width = "100%";
            exportBtn.disabled = false;
            
            // Inicia predição se webcam ok
            if (isVideoReady) {
                predictBtn.style.display = 'block';
                predictBtn.innerText = "Pausar Previsão";
                isPredicting = true;
                predict();
            } else {
                 predictBtn.style.display = 'block';
                 predictBtn.innerText = "Iniciar Previsão (Requer webcam)";
            }
            
            setupPredictionUI();
        }
    });
});

function setupPredictionUI() {
    predictionResults.innerHTML = '';
    Object.values(classesData).forEach(cls => {
        const wrapper = document.createElement('div');
        wrapper.className = 'prediction-item';
        wrapper.innerHTML = `
            <div class="prediction-label">
                <span>${cls.name}</span>
                <span id="conf-${sanitizeId(cls.name)}">0%</span>
            </div>
            <div class="prediction-bar-bg">
                <div class="prediction-bar-fill" id="bar-${sanitizeId(cls.name)}" style="width: 0%"></div>
            </div>
        `;
        predictionResults.appendChild(wrapper);
    });
}

function sanitizeId(str) {
    return str.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
}

function predict() {
    if (!isPredicting) return;
    
    classifier.classify(webcamVideo, (err, results) => {
        if (err) {
            console.error(err);
            return;
        }

        results.forEach(res => {
            const cleanId = sanitizeId(res.label);
            const confEl = document.getElementById(`conf-${cleanId}`);
            const barEl = document.getElementById(`bar-${cleanId}`);
            
            if (confEl && barEl) {
                const confPercent = (res.confidence * 100).toFixed(1) + '%';
                confEl.innerText = confPercent;
                barEl.style.width = confPercent;
                
                // Muda a cor da barra dependendo da confiança
                if (res.confidence > 0.8) {
                    barEl.style.background = 'var(--secondary)'; // verde
                } else if (res.confidence > 0.4) {
                    barEl.style.background = 'var(--primary)'; // azul
                } else {
                    barEl.style.background = 'var(--danger)'; // vermelho
                }
            }
        });

        if (isPredicting) {
            requestAnimationFrame(predict);
        }
    });
}

// Predict Button Trigger
predictBtn.addEventListener('click', () => {
    isPredicting = !isPredicting;
    if (isPredicting) {
        predictBtn.innerText = "Pausar Previsão";
        predict();
    } else {
        predictBtn.innerText = "Retomar Previsão";
    }
});

// Export Model
exportBtn.addEventListener('click', () => {
    classifier.save();
});

// Add Class Button
addClassBtn.addEventListener('click', () => addClass());

// Boot
init();
