const upload = document.getElementById('upload');
const previewCanvas = document.getElementById('previewCanvas');
const ctx = previewCanvas.getContext('2d');

let originalImage = null;

upload.addEventListener('change', loadImage);

function loadImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
        originalImage = img;
        updatePreview();
    };
}

function updatePreview() {
    if (!originalImage) return;

    const filter = document.getElementById('filter').value;
    const scale = document.getElementById('scale').value / 100;
    const textInput = document.getElementById('textInput').value;
    const fontSize = document.getElementById('fontSize').value;
    const textColor = document.getElementById('textColor').value;
    const textX = document.getElementById('textX').value;
    const textY = document.getElementById('textY').value;

    // Set canvas size
    previewCanvas.width = originalImage.width * scale;
    previewCanvas.height = originalImage.height * scale;

    // Draw scaled image
    ctx.drawImage(originalImage, 0, 0, previewCanvas.width, previewCanvas.height);

    // Apply filter
    if (filter !== 'none') {
        const imageData = ctx.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
        applyFilter(imageData, filter);
        ctx.putImageData(imageData, 0, 0);
    }

    // Add text
    if (textInput) {
        ctx.font = `${fontSize}px Arial`;
        ctx.fillStyle = textColor;
        ctx.fillText(textInput, parseInt(textX), parseInt(textY));
    }

    // Update download link
    const downloadLink = document.getElementById('downloadLink');
    downloadLink.href = previewCanvas.toDataURL('image/jpeg');
    downloadLink.style.display = 'block';
    downloadLink.innerText = 'Download Processed Image';
}

function applyFilter(imageData, filterType) {
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        switch (filterType) {
            case 'grayscale':
                const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                data[i] = data[i + 1] = data[i + 2] = gray;
                break;
            case 'sepia':
                data[i] = Math.min(255, r * 0.393 + g * 0.769 + b * 0.189);
                data[i + 1] = Math.min(255, r * 0.349 + g * 0.686 + b * 0.168);
                data[i + 2] = Math.min(255, r * 0.272 + g * 0.534 + b * 0.131);
                break;
            case 'invert':
                data[i] = 255 - r;
                data[i + 1] = 255 - g;
                data[i + 2] = 255 - b;
                break;
        }
    }
}

function processImage() {
    if (!originalImage) return;

    const quality = document.getElementById('quality').value;

    // Get the current canvas image data
    const imageData = previewCanvas.toDataURL('image/jpeg', quality / 100);

    // Send to Python backend for final processing
    fetch('http://localhost:5000/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData,
            quality: quality
        }),
    })
    .then(response => response.json())
    .then(data => {
        const downloadLink = document.getElementById('downloadLink');
        downloadLink.href = data.processedImage;
        downloadLink.style.display = 'block';
        downloadLink.innerText = 'Download Processed Image';
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}