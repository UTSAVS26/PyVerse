// Configuration
const API_BASE_URL = "http://localhost:8000";
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Initialize context menus
chrome.runtime.onInstalled.addListener(() => {
  initContextMenus();
  checkServerStatus();
});

// Initialize context menus
function initContextMenus() {
  chrome.contextMenus.removeAll(() => {
    // Image context menu
    chrome.contextMenus.create({
      id: "detect-image",
      title: "🔍 Detect Image Deepfake",
      contexts: ["image"]
    });

    // Audio context menu
    chrome.contextMenus.create({
      id: "detect-audio",
      title: "🔍 Detect Audio Deepfake",
      contexts: ["audio"]
    });

    // Video context menu
    chrome.contextMenus.create({
      id: "detect-video",
      title: "🔍 Detect Video Deepfake",
      contexts: ["video"]
    });
  });
}

// Check server status
async function checkServerStatus() {
  try {
    const response = await fetchWithRetry(`${API_BASE_URL}/health`);
    const data = await response.json();
    
    if (!data.models_loaded) {
      showNotification('Warning', 'Models not fully loaded. Detection may be unavailable.');
    }
  } catch (error) {
    console.warn('Server not responding:', error.message);
  }
}

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!info.srcUrl) {
    showNotification('Error', 'No media source available');
    return;
  }

  try {
    showNotification('Processing', 'Analyzing media for deepfake...');

    const endpointMap = {
      "detect-image": "/detect-image",
      "detect-audio": "/detect-audio",
      "detect-video": "/detect-video"
    };

    const endpoint = endpointMap[info.menuItemId];
    if (!endpoint) return;

    // Fetch and process media
    const response = await fetch(info.srcUrl);
    if (!response.ok) throw new Error('Failed to fetch media');
    
    const blob = await response.blob();
    const base64Data = await blobToBase64(blob);
    
    // Determine payload based on media type
    const mediaType = info.menuItemId.split('-')[1];
    const payload = { [mediaType]: base64Data };

    // Send to detection API with retry logic
    const result = await fetchWithRetry(
      `${API_BASE_URL}${endpoint}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      },
      MAX_RETRIES
    );

    showNotification('Result', `This content is: ${result.result} (${(result.confidence * 100).toFixed(1)}%)`);

  } catch (error) {
    console.error('Detection failed:', error);
    showNotification('Error', error.message || 'Detection failed');
  }
});

// Fetch with retry logic
async function fetchWithRetry(url, options = {}, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (i + 1)));
    }
  }
}

// Helper functions
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function showNotification(title, message) {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon48.png',
    title: title,
    message: message,
    priority: 2
  });
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    fetchWithRetry,
    blobToBase64
  };
}