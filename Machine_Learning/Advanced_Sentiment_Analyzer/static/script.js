// Character counter functionality
document.addEventListener('DOMContentLoaded', function() {
    const textArea = document.getElementById('textToAnalyze');
    const charCount = document.getElementById('charCount');
    
    textArea.addEventListener('input', function() {
        const count = this.value.length;
        charCount.textContent = count;
        
        // Color coding for character count
        if (count > 500) {
            charCount.style.color = '#e17055';
        } else if (count > 300) {
            charCount.style.color = '#fdcb6e';
        } else {
            charCount.style.color = '#666';
        }
    });
    
    // Auto-resize textarea
    textArea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
});

async function analyzeSentiment() {
    const textToAnalyze = document.getElementById('textToAnalyze').value.trim();
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    // Hide previous results and errors
    results.style.display = 'none';
    error.style.display = 'none';
    
    // Validate input
    if (!textToAnalyze) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (textToAnalyze.length < 5) {
        showError('Please enter at least 5 characters for a meaningful analysis.');
        return;
    }
    
    // Show loading state
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('textToAnalyze', textToAnalyze);
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed. Please try again.');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (err) {
        showError(err.message || 'An error occurred during analysis. Please try again.');
    } finally {
        // Hide loading state
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    const results = document.getElementById('results');
    const sentimentBadge = document.getElementById('sentimentBadge');
    const sentimentIcon = document.getElementById('sentimentIcon');
    const sentimentLabel = document.getElementById('sentimentLabel');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceScore = document.getElementById('confidenceScore');
    
    // Determine sentiment display
    const isPositive = data.label === 'POSITIVE';
    const isNegative = data.label === 'NEGATIVE';
    const isNeutral = data.label === 'NEUTRAL';
    
    let sentimentText, iconClass, badgeClass;
    
    if (isPositive) {
        sentimentText = 'Positive';
        iconClass = 'fas fa-smile';
        badgeClass = 'positive';
    } else if (isNegative) {
        sentimentText = 'Negative';
        iconClass = 'fas fa-frown';
        badgeClass = 'negative';
    } else {
        sentimentText = 'Neutral';
        iconClass = 'fas fa-meh';
        badgeClass = 'neutral';
    }
    
    // Update sentiment badge
    sentimentBadge.className = `sentiment-badge ${badgeClass}`;
    sentimentIcon.className = `sentiment-icon ${iconClass}`;
    sentimentLabel.textContent = sentimentText;
    
    // Update confidence score
    const confidence = Math.round(data.score * 100);
    confidenceScore.textContent = confidence;
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    // Show results with animation
    results.style.display = 'block';
    
    // Add celebration effect for high confidence
    if (confidence > 90) {
        addCelebrationEffect();
    }
}

function showError(message) {
    const error = document.getElementById('error');
    const errorText = document.getElementById('errorText');
    
    errorText.textContent = message;
    error.style.display = 'flex';
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        error.style.display = 'none';
    }, 5000);
}

function addCelebrationEffect() {
    // Add a subtle celebration animation
    const card = document.querySelector('.analysis-card');
    card.style.animation = 'celebration 0.6s ease-in-out';
    
    setTimeout(() => {
        card.style.animation = '';
    }, 600);
}

// Add celebration keyframe animation
const style = document.createElement('style');
style.textContent = `
    @keyframes celebration {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
`;
document.head.appendChild(style);

// Enable Enter key to submit
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
        analyzeSentiment();
    }
});

// Add sample texts functionality
function addSampleTexts() {
    const samples = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The weather today is quite nice and pleasant.",
        "I'm feeling frustrated with this situation.",
        "Thank you so much for your help! You're wonderful.",
        "This movie was okay, nothing special but not bad either."
    ];
    
    // Add sample text buttons (optional enhancement)
    const buttonSection = document.querySelector('.button-section');
    const sampleContainer = document.createElement('div');
    sampleContainer.className = 'sample-container';
    const paragraph = document.createElement('p');
    paragraph.style.cssText = 'margin-bottom: 1rem; color: #666; font-size: 0.9rem;';
    paragraph.textContent = 'Try these sample texts:';
    
    const buttonsDiv = document.createElement('div');
    buttonsDiv.className = 'sample-buttons';
    
    samples.forEach(sample => {
        const button = document.createElement('button');
        button.className = 'sample-btn';
        button.textContent = sample.substring(0, 30) + '...';
        button.addEventListener('click', () => setSampleText(sample));
        buttonsDiv.appendChild(button);
    });
    
    sampleContainer.appendChild(paragraph);
    sampleContainer.appendChild(buttonsDiv);

    
    buttonSection.appendChild(sampleContainer);
}

function setSampleText(text) {
    const textArea = document.getElementById('textToAnalyze');
    textArea.value = text;
    textArea.dispatchEvent(new Event('input')); // Trigger char counter update
    textArea.focus();
}

// Initialize sample texts on page load
document.addEventListener('DOMContentLoaded', addSampleTexts);
