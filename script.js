// Global variables
let currentAudio = null;
let isPlaying = false;
let audioFile = null;

// Recording variables
let recorder = null;
let isRecording = false;
let recordingInterval = null;
let recordingStartTime = null;

// DOM elements
const uploadArea = document.getElementById("uploadArea");
const audioFileInput = document.getElementById("audioFile");
const audioPlayer = document.getElementById("audioPlayer");
const playBtn = document.getElementById("playBtn");
const fileName = document.getElementById("fileName");
const progressFill = document.getElementById("progressFill");
const currentTimeEl = document.getElementById("currentTime");
const durationEl = document.getElementById("duration");
const analyzeBtn = document.getElementById("analyzeBtn");
const results = document.getElementById("results");
const loadingOverlay = document.getElementById("loadingOverlay");

// Recording DOM elements
const recordBtn = document.getElementById("recordBtn");
const recordText = document.getElementById("recordText");
const recordingStatus = document.getElementById("recordingStatus");
const recordingTimerEl = document.getElementById("recordingTimer");

// Emotion data
const emotionData = {
  happy: {
    name: "Happy",
    description:
      "Joy, excitement, and positive emotions detected in your audio.",
    color: "#fbbf24",
    icon: "fas fa-smile",
  },
  sad: {
    name: "Sad",
    description:
      "Sadness, grief, and melancholic feelings detected in your audio.",
    color: "#3b82f6",
    icon: "fas fa-frown",
  },
  angry: {
    name: "Angry",
    description:
      "Anger, frustration, and aggressive tones detected in your audio.",
    color: "#ef4444",
    icon: "fas fa-angry",
  },
  fear: {
    name: "Fear",
    description:
      "Anxiety, worry, and fearful expressions detected in your audio.",
    color: "#8b5cf6",
    icon: "fas fa-surprise",
  },
  disgust: {
    name: "Disgust",
    description:
      "Revulsion, contempt, and negative reactions detected in your audio.",
    color: "#10b981",
    icon: "fas fa-meh",
  },
  neutral: {
    name: "Neutral",
    description:
      "Calm, balanced, and emotionally neutral tone detected in your audio.",
    color: "#6b7280",
    icon: "fas fa-minus",
  },
};

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeNavigation();
  initializeUpload();
  initializeAudioPlayer();
  initializeAnalyzeButton();
  initializeRecording();
});

// Navigation functionality
function initializeNavigation() {
  const navLinks = document.querySelectorAll(".nav-link");

  navLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href").substring(1);
      scrollToSection(targetId);

      // Update active nav link
      navLinks.forEach((l) => l.classList.remove("active"));
      this.classList.add("active");
    });
  });

  // Update active nav link on scroll
  window.addEventListener("scroll", updateActiveNavLink);
}

function scrollToSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    const navHeight = document.querySelector(".navbar").offsetHeight;
    const sectionTop = section.offsetTop - navHeight - 20;

    window.scrollTo({
      top: sectionTop,
      behavior: "smooth",
    });
  }
}

function updateActiveNavLink() {
  const sections = ["home", "about", "demo", "classes"];
  const navHeight = document.querySelector(".navbar").offsetHeight;

  let currentSection = "";
  sections.forEach((sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      const sectionTop = section.offsetTop - navHeight - 100;
      const sectionBottom = sectionTop + section.offsetHeight;

      if (window.scrollY >= sectionTop && window.scrollY < sectionBottom) {
        currentSection = sectionId;
      }
    }
  });

  // Update active nav link
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach((link) => {
    link.classList.remove("active");
    if (link.getAttribute("href") === `#${currentSection}`) {
      link.classList.add("active");
    }
  });
}

// Upload functionality
function initializeUpload() {
  // Click to upload
  uploadArea.addEventListener("click", () => {
    audioFileInput.click();
  });

  // File input change
  audioFileInput.addEventListener("change", handleFileSelect);

  // Drag and drop
  uploadArea.addEventListener("dragover", handleDragOver);
  uploadArea.addEventListener("dragleave", handleDragLeave);
  uploadArea.addEventListener("drop", handleDrop);
}

function handleDragOver(e) {
  e.preventDefault();
  uploadArea.classList.add("dragover");
}

function handleDragLeave(e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
}

function handleDrop(e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileSelect({ target: { files: files } });
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  // Validate file type
  if (!file.type.startsWith("audio/")) {
    alert("Please select an audio file.");
    return;
  }

  audioFile = file;
  loadAudioFile(file);
}

function loadAudioFile(file) {
  const url = URL.createObjectURL(file);
  currentAudio = new Audio(url);

  // Update UI
  fileName.textContent = file.name;
  audioPlayer.style.display = "block";
  analyzeBtn.disabled = false;
  results.style.display = "none";

  // Setup audio event listeners
  currentAudio.addEventListener("loadedmetadata", updateDuration);
  currentAudio.addEventListener("timeupdate", updateProgress);
  currentAudio.addEventListener("ended", handleAudioEnd);

  // Update upload area text
  uploadArea.innerHTML = `
        <div class="upload-content">
            <div class="upload-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h3>File loaded successfully!</h3>
            <p>Click to select a different file</p>
            <p class="upload-formats">Current: ${file.name}</p>
        </div>
    `;
}

// Audio player functionality
function initializeAudioPlayer() {
  playBtn.addEventListener("click", togglePlayPause);
}

function togglePlayPause() {
  if (!currentAudio) return;

  if (isPlaying) {
    currentAudio.pause();
    playBtn.innerHTML = '<i class="fas fa-play"></i>';
    isPlaying = false;
  } else {
    currentAudio.play();
    playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    isPlaying = true;
  }
}

function updateDuration() {
  if (currentAudio) {
    durationEl.textContent = formatTime(currentAudio.duration);
  }
}

function updateProgress() {
  if (currentAudio) {
    const progress = (currentAudio.currentTime / currentAudio.duration) * 100;
    progressFill.style.width = `${progress}%`;
    currentTimeEl.textContent = formatTime(currentAudio.currentTime);
  }
}

function handleAudioEnd() {
  playBtn.innerHTML = '<i class="fas fa-play"></i>';
  isPlaying = false;
  currentAudio.currentTime = 0;
  progressFill.style.width = "0%";
  currentTimeEl.textContent = "0:00";
}

function formatTime(seconds) {
  if (isNaN(seconds)) return "0:00";

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

// Analysis functionality
function initializeAnalyzeButton() {
  analyzeBtn.addEventListener("click", analyzeAudio);
}

async function analyzeAudio() {
  if (!audioFile) {
    alert("Please select an audio file first.");
    return;
  }

  showLoading(true);
  console.log("Starting audio analysis...");
  console.log(
    "Audio file:",
    audioFile.name,
    "Size:",
    audioFile.size,
    "Type:",
    audioFile.type
  );

  try {
    const formData = new FormData();
    formData.append("file", audioFile);

    console.log("Sending request to /predict...");

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", response.status);
    console.log("Response headers:", response.headers);

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Server error response:", errorText);
      throw new Error(`Server error: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    console.log("Analysis result:", result);

    if (result.error) {
      throw new Error(result.error);
    }

    displayResults(result);
  } catch (error) {
    console.error("Error analyzing audio:", error);

    // Provide more specific error messages
    let errorMessage = "Failed to analyze audio. ";
    if (error.message.includes("Failed to fetch")) {
      errorMessage +=
        "Cannot connect to the server. Make sure the backend is running on port 8000.";
    } else if (error.message.includes("Server error")) {
      errorMessage += `Server error: ${error.message}`;
    } else {
      errorMessage += error.message;
    }

    displayError(errorMessage);
  } finally {
    showLoading(false);
  }
}

function displayResults(result) {
  if (result.error) {
    displayError(result.error);
    return;
  }

  // Map backend emotion to frontend emotion data
  const emotionKey = result.emotion.toLowerCase();
  const emotion = emotionData[emotionKey] || emotionData.neutral;

  // Update results UI
  document.getElementById("confidenceScore").textContent = `${
    result.confidence || 0
  }%`;
  document.getElementById("emotionName").textContent = emotion.name;
  document.getElementById("emotionDescription").textContent =
    emotion.description;

  // Update emotion icon
  const emotionIcon = document.getElementById("emotionIconLarge");
  emotionIcon.innerHTML = `<i class="${emotion.icon}"></i>`;
  emotionIcon.style.background = emotion.color;

  // Update probability breakdown
  updateProbabilityBars(result.all_probabilities || {});

  // Show results
  results.style.display = "block";

  // Scroll to results
  setTimeout(() => {
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

function updateProbabilityBars(probabilities) {
  const probabilityBars = document.getElementById("probabilityBars");
  probabilityBars.innerHTML = "";

  // Sort emotions by probability
  const sortedEmotions = Object.entries(probabilities)
    .map(([emotion, prob]) => ({
      emotion: emotion.toLowerCase(),
      probability: prob,
    }))
    .sort((a, b) => b.probability - a.probability);

  sortedEmotions.forEach(({ emotion, probability }) => {
    const emotionInfo = emotionData[emotion] || emotionData.neutral;

    const probBar = document.createElement("div");
    probBar.className = "prob-bar";
    probBar.innerHTML = `
            <div class="prob-label">${emotionInfo.name}</div>
            <div class="prob-bar-fill">
                <div class="prob-fill" style="width: 0%; background: ${emotionInfo.color}"></div>
            </div>
            <div class="prob-value">${probability}%</div>
        `;

    probabilityBars.appendChild(probBar);

    // Animate the bar
    setTimeout(() => {
      const fill = probBar.querySelector(".prob-fill");
      fill.style.width = `${probability}%`;
    }, 100);
  });
}

function displayError(message) {
  results.innerHTML = `
        <div class="error-message" style="text-align: center; padding: 2rem; color: var(--error-color);">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
            <h3>Error</h3>
            <p>${message}</p>
        </div>
    `;
  results.style.display = "block";
}

function showLoading(show) {
  if (show) {
    loadingOverlay.classList.add("active");
  } else {
    loadingOverlay.classList.remove("active");
  }
}

// Utility functions
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Add some interactive animations
document.addEventListener("DOMContentLoaded", function () {
  // Animate emotion cards on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.animation = "fadeInUp 0.6s ease forwards";
      }
    });
  }, observerOptions);

  // Observe elements for animation
  document.querySelectorAll(".emotion-card, .step").forEach((el) => {
    observer.observe(el);
  });

  // Add CSS for fadeInUp animation
  const style = document.createElement("style");
  style.textContent = `
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .emotion-card, .step {
            opacity: 0;
        }
    `;
  document.head.appendChild(style);
});

// Add smooth scrolling for hero buttons
document.querySelectorAll(".hero-buttons .btn").forEach((btn) => {
  btn.addEventListener("click", function (e) {
    if (this.textContent.includes("Try Demo")) {
      e.preventDefault();
      scrollToSection("demo");
    } else if (this.textContent.includes("Learn More")) {
      e.preventDefault();
      scrollToSection("about");
    }
  });
});

// Add hover effects for emotion cards
document.querySelectorAll(".emotion-card").forEach((card) => {
  card.addEventListener("mouseenter", function () {
    this.style.transform = "translateY(-10px) scale(1.02)";
  });

  card.addEventListener("mouseleave", function () {
    this.style.transform = "translateY(0) scale(1)";
  });
});

// Add click handlers for emotion cards to scroll to demo
document.querySelectorAll(".emotion-card").forEach((card) => {
  card.addEventListener("click", function () {
    scrollToSection("demo");
  });
});

// Recording functionality
function initializeRecording() {
  recordBtn.addEventListener("click", toggleRecording);
}

async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    console.log("Requesting microphone access...");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    recorder = new RecordRTC(stream, {
      type: "audio",
      mimeType: "audio/wav",
      recorderType: RecordRTC.StereoAudioRecorder,
      desiredSampRate: 16000,
      numberOfAudioChannels: 1,
      timeSlice: 1000, // Record in 1-second chunks
    });

    recorder.startRecording();
    isRecording = true;
    recordingStartTime = Date.now();

    // Update UI
    recordBtn.classList.add("recording");
    recordBtn.innerHTML =
      '<i class="fas fa-stop"></i><span>Stop Recording</span>';
    recordText.textContent = "Stop Recording";
    recordingStatus.style.display = "block";

    // Start timer
    startRecordingTimer();

    console.log("Recording started successfully");
  } catch (error) {
    console.error("Error starting recording:", error);
    alert(
      "Unable to access microphone. Please check your permissions and try again."
    );
  }
}

function stopRecording() {
  if (!recorder || !isRecording) return;

  console.log("Stopping recording...");

  recorder.stopRecording(async () => {
    try {
      const blob = recorder.getBlob();
      console.log("Recording completed, blob size:", blob.size);

      // Reset recording state
      isRecording = false;
      if (recordingInterval) {
        clearInterval(recordingInterval);
        recordingInterval = null;
      }

      // Update UI
      recordBtn.classList.remove("recording");
      recordBtn.innerHTML =
        '<i class="fas fa-microphone"></i><span>Start Recording</span>';
      recordText.textContent = "Start Recording";
      recordingStatus.style.display = "none";

      // Create audio file from blob and set global variable
      audioFile = new File([blob], "recording.wav", {
        type: "audio/wav",
      });

      // Load the recorded audio
      loadAudioFile(audioFile);

      // Automatically analyze the recording
      setTimeout(() => {
        analyzeAudio();
      }, 500);

      console.log("Recording processed successfully");
    } catch (error) {
      console.error("Error processing recording:", error);
      alert("Error processing recording. Please try again.");
    }
  });
}

function startRecordingTimer() {
  recordingInterval = setInterval(() => {
    if (!recordingStartTime) return;

    const elapsed = Date.now() - recordingStartTime;
    const minutes = Math.floor(elapsed / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);

    const timeString = `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
    recordingTimerEl.textContent = timeString;

    // Auto-stop after 30 seconds
    if (elapsed >= 30000) {
      stopRecording();
    }
  }, 100);
}
