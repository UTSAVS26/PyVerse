// Content script for extracting media from web pages

// Get the best quality image from page
function getImage() {
  const images = Array.from(document.querySelectorAll('img'));
  if (images.length === 0) return null;
  
  // Filter out small or likely decorative images
  const contentImages = images.filter(img => {
    return img.naturalWidth >= 100 && 
           img.naturalHeight >= 100 &&
           !img.src.includes('pixel') &&
           !img.src.includes('track');
  });
  
  if (contentImages.length === 0) return null;
  
  // Select the largest image
  contentImages.sort((a, b) => 
    (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight)
  );
  
  const bestImage = contentImages[0];
  
  // Create canvas to get image data
  const canvas = document.createElement('canvas');
  canvas.width = bestImage.naturalWidth;
  canvas.height = bestImage.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bestImage, 0, 0);
  
  return canvas.toDataURL('image/jpeg');
}

// Get audio from page
function getAudio() {
  // Check for audio elements
  const audioElement = document.querySelector('audio');
  if (audioElement && audioElement.src) {
    return audioElement.src;
  }
  
  // Check for audio sources
  const audioSource = document.querySelector('audio source');
  if (audioSource && audioSource.src) {
    return audioSource.src;
  }
  
  // Check for video elements (which may contain audio)
  const videoElement = document.querySelector('video');
  if (videoElement && videoElement.src) {
    return videoElement.src;
  }
  
  // Check for video sources
  const videoSource = document.querySelector('video source');
  if (videoSource && videoSource.src) {
    return videoSource.src;
  }
  
  return null;
}

// Get video from page
function getVideo() {
  // Check for video elements
  const videoElement = document.querySelector('video');
  if (videoElement && videoElement.src) {
    return videoElement.src;
  }
  
  // Check for video sources
  const videoSource = document.querySelector('video source');
  if (videoSource && videoSource.src) {
    return videoSource.src;
  }
  
  // Check for embedded videos (YouTube, Vimeo, etc.)
  const iframes = document.querySelectorAll('iframe');
  for (const iframe of iframes) {
    const src = iframe.src || '';
    if (src.includes('youtube.com/embed/') || 
        src.includes('youtu.be/') ||
        src.includes('vimeo.com/') ||
        src.includes('dailymotion.com/embed/')) {
      return src;
    }
  }
  
  return null;
}

// Listen for messages from popup or background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  let result = null;
  
  switch(request.action) {
    case 'getImage':
      result = getImage();
      break;
    case 'getAudio':
      result = getAudio();
      break;
    case 'getVideo':
      result = getVideo();
      break;
  }
  
  sendResponse(result);
  return true; // Indicates we wish to send a response asynchronously
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    getImage,
    getAudio,
    getVideo
  };
}