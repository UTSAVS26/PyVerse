// Constants
const API_BASE_URL = "http://localhost:8000";
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

// UI elements
const elements = {
  image: {
    button: document.getElementById("detectImg"),
    result: document.getElementById("imgRes")
  },
  audio: {
    button: document.getElementById("startAudio"),
    uploadButton: document.getElementById("uploadAudio"),
    input: document.getElementById("audioInput"),
    result: document.getElementById("audioRes")
  },
  video: {
    button: document.getElementById("detectVid"),
    uploadButton: document.getElementById("uploadVid"),
    input: document.getElementById("videoInput"),
    result: document.getElementById("vidRes"),
    uploadResult: document.getElementById("uploadRes")
  },
  status: document.getElementById("status")
};

// Initialize
function init() {
  // Check server status
  checkServerStatus();
  
  // Set up event listeners
  setupEventListeners();
}

// Check server status
async function checkServerStatus() {
  try {
    elements.status.textContent = "Checking server status...";
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    
    if (response.ok) {
      if (data.models_loaded) {
        elements.status.textContent = "Server connected ✓";
        elements.status.className = "status connected";
      } else {
        elements.status.textContent = "Server connected but models not loaded";
        elements.status.className = "status warning";
      }
    } else {
      elements.status.textContent = "Server error";
      elements.status.className = "status error";
    }
  } catch (error) {
    elements.status.textContent = "Server not connected";
    elements.status.className = "status error";
    disableButtons();
  }
}

// Disable buttons when server is not available
function disableButtons() {
  const buttons = document.querySelectorAll("button");
  buttons.forEach(button => {
    if (button.id !== "reconnect") {
      button.disabled = true;
    }
  });
}

// Set up event listeners
function setupEventListeners() {
  // Image detection
  elements.image.button.addEventListener("click", detectPageImage);
  
  // Audio detection
  elements.audio.button.addEventListener("click", detectPageAudio);
  elements.audio.uploadButton.addEventListener("click", handleAudioUpload);
  elements.audio.input.addEventListener("change", validateFileSize);
  
  // Video detection
  elements.video.button.addEventListener("click", detectPageVideo);
  elements.video.uploadButton.addEventListener("click", handleVideoUpload);
  elements.video.input.addEventListener("change", validateFileSize);
  
  // Reconnect button
  document.getElementById("reconnect").addEventListener("click", checkServerStatus);
}

// Validate file size
function validateFileSize(event) {
  const file = event.target.files[0];
  if (file && file.size > MAX_FILE_SIZE) {
    alert(`File is too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB`);
    event.target.value = "";
  }
}

// Image detection from page
async function detectPageImage() {
  setLoadingState(elements.image.result, true);
  
  try {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    const imageUrl = await chrome.tabs.sendMessage(tab.id, {action: "getImage"});
    
    if (!imageUrl) {
      setResult(elements.image.result, "No image found on page", "error");
      return;
    }
    
    const result = await analyzeMedia("image", imageUrl);
    setResult(elements.image.result, `${result.result} (${(result.confidence * 100).toFixed(1)}%)`, result.result.toLowerCase());
  } catch (error) {
    setResult(elements.image.result, error.message, "error");
  }
}

// Audio detection from page
async function detectPageAudio() {
  setLoadingState(elements.audio.result, true);
  
  try {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    const audioUrl = await chrome.tabs.sendMessage(tab.id, {action: "getAudio"});
    
    if (!audioUrl) {
      setResult(elements.audio.result, "No audio found on page", "error");
      return;
    }
    
    const result = await analyzeMedia("audio", audioUrl);
    setResult(elements.audio.result, `${result.result} (${(result.confidence * 100).toFixed(1)}%)`, result.result.toLowerCase());
  } catch (error) {
    setResult(elements.audio.result, error.message, "error");
  }
}

// Video detection from page
async function detectPageVideo() {
  setLoadingState(elements.video.result, true);
  
  try {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    const videoUrl = await chrome.tabs.sendMessage(tab.id, {action: "getVideo"});
    
    if (!videoUrl) {
      setResult(elements.video.result, "No video found on page", "error");
      return;
    }
    
    const result = await analyzeMedia("video", videoUrl);
    setResult(elements.video.result, `${result.result} (${(result.confidence * 100).toFixed(1)}%)`, result.result.toLowerCase());
  } catch (error) {
    setResult(elements.video.result, error.message, "error");
  }
}

// Handle audio upload
async function handleAudioUpload() {
  setLoadingState(elements.audio.result, true);
  const file = elements.audio.input.files[0];
  
  if (!file) {
    setResult(elements.audio.result, "Please select a file first", "error");
    return;
  }
  
  try {
    const base64Data = await fileToBase64(file);
    const result = await analyzeMedia("audio", base64Data);
    setResult(elements.audio.result, `${result.result} (${(result.confidence * 100).toFixed(1)}%)`, result.result.toLowerCase());
  } catch (error) {
    setResult(elements.audio.result, error.message, "error");
  }
}

// Handle video upload
async function handleVideoUpload() {
  setLoadingState(elements.video.uploadResult, true);
  const file = elements.video.input.files[0];
  
  if (!file) {
    setResult(elements.video.uploadResult, "Please select a file first", "error");
    return;
  }
  
  try {
    const base64Data = await fileToBase64(file);
    const result = await analyzeMedia("video", base64Data);
    setResult(elements.video.uploadResult, `${result.result} (${(result.confidence * 100).toFixed(1)}%)`, result.result.toLowerCase());
  } catch (error) {
    setResult(elements.video.uploadResult, error.message, "error");
  }
}

// Analyze media through API
async function analyzeMedia(type, urlOrData) {
  let base64Data;
  
  if (urlOrData.startsWith('http')) {
    const response = await fetch(urlOrData);
    if (!response.ok) throw new Error('Failed to fetch media');
    const blob = await response.blob();
    base64Data = await blobToBase64(blob);
  } else {
    base64Data = urlOrData;
  }
  
  const apiResponse = await fetch(`${API_BASE_URL}/detect-${type}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ [type]: base64Data })
  });
  
  if (!apiResponse.ok) {
    const error = await apiResponse.json();
    throw new Error(error.detail || 'API request failed');
  }
  
  return await apiResponse.json();
}

// UI helper functions
function setLoadingState(element, isLoading) {
  if (isLoading) {
    element.textContent = "⏳ Processing...";
    element.className = "result loading";
  }
}

function setResult(element, message, type) {
  element.textContent = message;
  element.className = `result ${type}`;
  
  // Add appropriate icon based on type
  const icon = type === "real" ? "✅" : 
               type === "fake" ? "❌" : 
               "⚠️";
  element.textContent = `${icon} ${message}`;
}

// Helper functions
async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);