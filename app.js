// app.js
// Main application controller integrating webcam/screen share, Web Audio API, MediaPipe FaceMesh, Chart.js, and tracker.js

// State Variables
let monitoringActive = false;
let cameraStream = null;
let audioContext = null;
let audioStream = null;
let mediaPipeCamera = null;
let faceMesh = null;
let chartInterval = null;
let sessionStartTime = null;

// Screen Capture ROI Selector State
let cropROI = { x: 0, y: 0, w: 0, h: 0 };
let isSelectingROI = false;
let isDrawingROI = false;
let roiStart = { x: 0, y: 0 };
let roiEnd = { x: 0, y: 0 };
let selectedBox = null; // { x, y, w, h } in canvas coordinates
let isProcessingFrame = false; // Queue lock to prevent overlapping faceMesh runs

// Tracker Class Instances
let blinkAnalyzer = null;
let gazeTracker = null;
let headPoseEstimator = null;
let integrityMonitor = null;
let emotionInference = null;
let scoringEngine = null;
let timelineAnalytics = null;

// Real-time State Snapshot
let currentRealtimeState = { attention: 1.0, confidence: 1.0, focus: 1.0, stress: 0.0, speech_energy: 0.0, feeling: "Neutral" };
let currentRawStats = { gaze: "Center", head_pose: "Center", speaking: false, integrity_flags: [] };
let currentEnvMetrics = { lighting_ok: true, blur_ok: true, landmarks_stable: true, face_visible: true, roi_stable: true };

// History Arrays for Post-Session Charting
let chartHistoryTime = [];
let chartHistoryAttention = [];
let chartHistoryFocus = [];
let chartHistoryConfidence = [];
let chartHistoryStress = [];

// Jitter and ROI Drift Queues
const jitterHistory = [];
const faceWidthHistory = [];
let prevNosePos = null;

// Chart.js Instances
let attentionChart = null;
let communicationChart = null;

// DOM Elements
const videoEl = document.getElementById("webcam-video");
const canvasEl = document.getElementById("webcam-canvas");
const ctx = canvasEl.getContext("2d");
const systemDot = document.getElementById("system-dot");
const systemText = document.getElementById("system-status-text");

const captureMethodSelect = document.getElementById("capture-method");
const cameraSelectGroup = document.getElementById("camera-select-group");
const cameraSelect = document.getElementById("camera-select");
const micSelect = document.getElementById("mic-select");
const roiControlsGroup = document.getElementById("roi-controls-group");
const btnResetROI = document.getElementById("btn-reset-roi");

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");

// ROI Overlay Prompt Elements
const roiOverlay = document.getElementById("roi-overlay");
const btnROIConfirm = document.getElementById("btn-roi-confirm");
const btnROIReselect = document.getElementById("btn-roi-reselect");
const faceMeshStatusBadge = document.getElementById("facemesh-status-badge");

const qualityLighting = document.getElementById("quality-lighting");
const qualityBlur = document.getElementById("quality-blur");
const qualityJitter = document.getElementById("quality-jitter");
const qualityRoi = document.getElementById("quality-roi");

const logConsole = document.getElementById("log-console");
const logEmptyState = document.getElementById("log-empty-state");
const logCounter = document.getElementById("log-counter");

// Gauge Value Elements
const valScore = document.getElementById("val-score");
const valAttention = document.getElementById("val-attention");
const valCommunication = document.getElementById("val-communication");
const valIntegrity = document.getElementById("val-integrity");
const valStress = document.getElementById("val-stress");
const valInferenceConf = document.getElementById("val-inference-conf");

// Gauge Circle Paths
const gaugeScore = document.getElementById("gauge-score");
const gaugeAttention = document.getElementById("gauge-attention");
const gaugeCommunication = document.getElementById("gauge-communication");
const gaugeIntegrity = document.getElementById("gauge-integrity");
const gaugeStress = document.getElementById("gauge-stress");
const gaugeInferenceConf = document.getElementById("gauge-inference-conf");

// Traceback Elements
const traceAttention = document.getElementById("trace-attention");
const traceFocus = document.getElementById("trace-focus");
const traceComm = document.getElementById("trace-comm");
const traceGaze = document.getElementById("trace-gaze");
const traceStress = document.getElementById("trace-stress");
const traceIntegrity = document.getElementById("trace-integrity");

// Initialize on Load
window.addEventListener("DOMContentLoaded", async () => {
    initRealtimeCharts();
    await enumerateDevices();
    setupEventListeners();
    setupReportFinalizeListeners();
    setupMediaPipe();
});

// Helper to get time string
function getCurrentTimeStr() {
    if (!sessionStartTime) return "00:00";
    const timestamp = (Date.now() - sessionStartTime) / 1000;
    const minutes = Math.floor(timestamp / 60);
    const seconds = Math.floor(timestamp % 60);
    return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

// Event Listener setup
function setupEventListeners() {
    btnStart.addEventListener("click", startMonitoring);
    btnStop.addEventListener("click", stopMonitoring);
    btnReset.addEventListener("click", resetDashboard);
    
    captureMethodSelect.addEventListener("change", () => {
        if (captureMethodSelect.value === "webcam") {
            cameraSelectGroup.style.display = "flex";
            roiControlsGroup.style.display = "none";
            cropROI = { x: 0, y: 0, w: 0, h: 0 };
            selectedBox = null;
        } else {
            cameraSelectGroup.style.display = "none";
            roiControlsGroup.style.display = "flex";
        }
        if (monitoringActive) {
            stopMonitoring();
        }
    });

    btnResetROI.addEventListener("click", () => {
        enterROIPromptSelection();
    });

    btnROIReselect.addEventListener("click", () => {
        selectedBox = null;
        cropROI = { x: 0, y: 0, w: 0, h: 0 };
        btnROIConfirm.disabled = true;
    });

    btnROIConfirm.addEventListener("click", () => {
        confirmROIPromptSelection();
    });

    cameraSelect.addEventListener("change", async () => {
        if (monitoringActive) {
            stopMonitoring();
            await startMonitoring();
        }
    });

    // Bounding Box Drag Selection (ROI Selection on Canvas)
    canvasEl.addEventListener("mousedown", (e) => {
        if (!monitoringActive || captureMethodSelect.value !== "screen" || !isSelectingROI) return;
        
        isDrawingROI = true;
        const rect = canvasEl.getBoundingClientRect();
        const scaleX = 640 / rect.width;
        const scaleY = 360 / rect.height;
        
        roiStart.x = (e.clientX - rect.left) * scaleX;
        roiStart.y = (e.clientY - rect.top) * scaleY;
        roiEnd.x = roiStart.x;
        roiEnd.y = roiStart.y;
        selectedBox = null;
    });

    canvasEl.addEventListener("mousemove", (e) => {
        if (!isDrawingROI) return;
        
        const rect = canvasEl.getBoundingClientRect();
        const scaleX = 640 / rect.width;
        const scaleY = 360 / rect.height;
        
        roiEnd.x = (e.clientX - rect.left) * scaleX;
        roiEnd.y = (e.clientY - rect.top) * scaleY;
    });

    canvasEl.addEventListener("mouseup", (e) => {
        if (!isDrawingROI) return;
        isDrawingROI = false;

        const x = Math.min(roiStart.x, roiEnd.x);
        const y = Math.min(roiStart.y, roiEnd.y);
        const w = Math.abs(roiStart.x - roiEnd.x);
        const h = Math.abs(roiStart.y - roiEnd.y);

        if (w > 20 && h > 20) {
            selectedBox = { x, y, w, h };
            btnROIConfirm.disabled = false;
        } else {
            selectedBox = null;
            btnROIConfirm.disabled = true;
        }
    });

    // Toast log listener
    window.addEventListener("timeline-event-logged", (e) => {
        const ev = e.detail;
        showToast(ev.type, ev.name, ev.reason_data.reason.join(", "));
        addConsoleLog(ev.time_str, ev.type, ev.name, ev.reason_data.reason.join(", "));
    });

    // Post session actions
    document.getElementById("btn-print").addEventListener("click", () => {
        window.print();
    });
    document.getElementById("btn-export-md").addEventListener("click", copyMarkdownReport);
}

// Device Enumeration
async function enumerateDevices() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then(stream => {
            stream.getTracks().forEach(track => track.stop());
        }).catch(() => {});

        const devices = await navigator.mediaDevices.enumerateDevices();
        
        cameraSelect.innerHTML = "";
        micSelect.innerHTML = "";
        
        let camerasCount = 0;
        let micsCount = 0;

        devices.forEach(device => {
            if (device.kind === "videoinput") {
                const opt = document.createElement("option");
                opt.value = device.deviceId;
                opt.textContent = device.label || `Webcam ${++camerasCount}`;
                cameraSelect.appendChild(opt);
            } else if (device.kind === "audioinput") {
                const opt = document.createElement("option");
                opt.value = device.deviceId;
                opt.textContent = device.label || `Microphone ${++micsCount}`;
                micSelect.appendChild(opt);
            }
        });

        if (cameraSelect.children.length === 0) {
            cameraSelect.innerHTML = '<option value="">No Webcam Found</option>';
        }
        if (micSelect.children.length === 0) {
            micSelect.innerHTML = '<option value="">No Microphone Found</option>';
        }
    } catch (err) {
        console.error("Device enumeration failed: ", err);
        cameraSelect.innerHTML = '<option value="">Permission Denied</option>';
        micSelect.innerHTML = '<option value="">Permission Denied</option>';
    }
}

// MediaPipe Setup
function setupMediaPipe() {
    faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 2, // Allow detecting multiple faces to trigger integrity warnings!
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onMeshResults);
}

// Initialize Realtime Chart.js plots
function initRealtimeCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 9 } }
            },
            y: {
                min: 0,
                max: 1.1,
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#9ca3af', font: { family: 'Outfit', size: 9 } }
            }
        },
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#f3f4f6', font: { family: 'Outfit', size: 10 }, boxWidth: 10 }
            }
        },
        animation: { duration: 200 }
    };

    const ctxAtt = document.getElementById("chart-attention").getContext("2d");
    attentionChart = new Chart(ctxAtt, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Attention (Gaze)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderWidth: 2, tension: 0.2, fill: false },
                { label: 'Focus (Head Pose)', data: [], borderColor: '#00b5d8', borderWidth: 1.5, borderDash: [4, 4], tension: 0.2, fill: false }
            ]
        },
        options: commonOptions
    });

    const ctxComm = document.getElementById("chart-communication").getContext("2d");
    communicationChart = new Chart(ctxComm, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Communication Confidence', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 2, tension: 0.2, fill: false },
                { label: 'Stress Indicators', data: [], borderColor: '#ef4444', borderWidth: 1.5, tension: 0.2, fill: false }
            ]
        },
        options: commonOptions
    });
}

// Consolidated Camera stream frame processing loop
function initCameraLoop() {
    if (mediaPipeCamera) return;

    mediaPipeCamera = new Camera(videoEl, {
        onFrame: async () => {
            if (!monitoringActive) return;

            // Constrain internal canvas size to 640x360 for light, high-performance MediaPipe runs
            const w = 640;
            const h = 360;
            canvasEl.width = w;
            canvasEl.height = h;

            if (isSelectingROI) {
                // ROI Selection state
                ctx.drawImage(videoEl, 0, 0, w, h);
                
                if (isDrawingROI) {
                    // Selection box outline
                    ctx.strokeStyle = "rgba(245, 158, 11, 0.9)";
                    ctx.lineWidth = 3;
                    ctx.setLineDash([6, 6]);
                    ctx.strokeRect(roiStart.x, roiStart.y, roiEnd.x - roiStart.x, roiEnd.y - roiStart.y);
                    ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
                    ctx.fillRect(roiStart.x, roiStart.y, roiEnd.x - roiStart.x, roiEnd.y - roiStart.y);
                    ctx.setLineDash([]);
                } else if (selectedBox) {
                    // Highlight selected box
                    ctx.strokeStyle = "var(--success)";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(selectedBox.x, selectedBox.y, selectedBox.w, selectedBox.h);
                    ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
                    ctx.fillRect(selectedBox.x, selectedBox.y, selectedBox.w, selectedBox.h);
                }
            } else {
                // Monitoring tracking state
                if (cropROI.w > 0 && cropROI.h > 0) {
                    // Draw only cropped sub-rectangle (fits 640x360 canvas)
                    ctx.drawImage(videoEl, cropROI.x, cropROI.y, cropROI.w, cropROI.h, 0, 0, w, h);
                } else {
                    ctx.drawImage(videoEl, 0, 0, w, h);
                }

                // Prevent overlapping asynchronous faceMesh.send triggers
                if (isProcessingFrame) return;
                isProcessingFrame = true;

                try {
                    await faceMesh.send({ image: canvasEl });
                } catch (err) {
                    console.error("MediaPipe FaceMesh run error: ", err);
                } finally {
                    isProcessingFrame = false;
                }
            }
        },
        width: 640,
        height: 360
    });

    mediaPipeCamera.start();
}

// Start Session
async function startMonitoring() {
    if (monitoringActive) return;

    try {
        btnStart.disabled = true;
        systemDot.className = "status-dot active";
        systemText.textContent = "Connecting Devices...";

        // Reset class instances
        blinkAnalyzer = new BlinkAnalysis();
        gazeTracker = new GazeTracking();
        headPoseEstimator = new HeadPoseEstimation();
        integrityMonitor = new IntegrityMonitoring();
        emotionInference = new EmotionInference();
        scoringEngine = new ScoringEngine();
        timelineAnalytics = new TimelineAnalytics();

        // Clear console and logs
        logConsole.innerHTML = "";
        logEmptyState.style.display = "none";
        logCounter.textContent = "0 events";

        // Reset chart arrays
        chartHistoryTime = [];
        chartHistoryAttention = [];
        chartHistoryFocus = [];
        chartHistoryConfidence = [];
        chartHistoryStress = [];

        // Reset UI View
        document.getElementById("dashboard-view").style.display = "flex";
        document.getElementById("report-view").style.display = "none";
        btnReset.style.display = "none";

        // Clear crop bounds
        cropROI = { x: 0, y: 0, w: 0, h: 0 };
        selectedBox = null;
        isProcessingFrame = false;

        // Start Microphone capture
        const audioId = micSelect.value;
        const audioConstraints = audioId ? { deviceId: { exact: audioId } } : true;
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
        startAudioProcessing(audioStream);

        // Define stream-loader handlers
        let metadataLoaded = false;
        const onMetadataLoaded = () => {
            if (metadataLoaded) return;
            metadataLoaded = true;

            videoEl.play();
            canvasEl.width = 640;
            canvasEl.height = 360;
            
            if (captureMethodSelect.value === "webcam") {
                isSelectingROI = false;
                monitoringActive = true;
                initCameraLoop();
                startFaceTrackingLoop();
            } else {
                enterROIPromptSelection();
            }
        };

        // Start Video (Webcam vs Screen Capture)
        if (captureMethodSelect.value === "webcam") {
            const cameraId = cameraSelect.value;
            const videoConstraints = cameraId ? { deviceId: { exact: cameraId }, width: 640, height: 360 } : { width: 640, height: 360 };
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
        } else {
            cameraStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    displaySurface: "window",
                    width: 1280,
                    height: 720
                }
            });
        }

        // Bulletproof listener assignment BEFORE loading srcObject
        videoEl.onloadedmetadata = onMetadataLoaded;
        videoEl.srcObject = cameraStream;

        // Fallback manually trigger after 1.2s in case event is swallowed by browser caching
        setTimeout(() => {
            if (!metadataLoaded && cameraStream) {
                console.log("[System] onloadedmetadata fallback triggered.");
                onMetadataLoaded();
            }
        }, 1200);

    } catch (err) {
        console.error("Could not access camera/screen/mic: ", err);
        btnStart.disabled = false;
        systemDot.className = "status-dot";
        systemText.textContent = "Error Accessing Devices";
        alert("Failed to start monitoring: Please ensure permissions are granted.");
    }
}

// Show ROI Prompt Selection Overlay
function enterROIPromptSelection() {
    isSelectingROI = true;
    selectedBox = null;
    cropROI = { x: 0, y: 0, w: 0, h: 0 };

    btnROIConfirm.disabled = true;
    roiOverlay.style.display = "flex";
    faceMeshStatusBadge.style.display = "none";

    systemText.textContent = "ROI Selection Required";
    systemDot.className = "status-dot active";
    btnStop.disabled = true;

    // Pause charts timer if active
    if (chartInterval) {
        clearInterval(chartInterval);
        chartInterval = null;
    }

    monitoringActive = true;
    initCameraLoop();
}

// Confirm Selection and Start Tracking
function confirmROIPromptSelection() {
    if (!selectedBox) return;

    // Map from 640x360 canvas coordinates to video source stream resolution
    const videoW = videoEl.videoWidth || 640;
    const videoH = videoEl.videoHeight || 360;
    
    const scaleX = videoW / 640;
    const scaleY = videoH / 360;

    cropROI = {
        x: selectedBox.x * scaleX,
        y: selectedBox.y * scaleY,
        w: selectedBox.w * scaleX,
        h: selectedBox.h * scaleY
    };

    roiOverlay.style.display = "none";
    faceMeshStatusBadge.style.display = "flex";
    isSelectingROI = false;

    // Trigger full tracking sequence
    startFaceTrackingLoop();
}

// Triggers active timers and logs
function startFaceTrackingLoop() {
    sessionStartTime = Date.now();
    timelineAnalytics.startTime = Date.now();

    systemDot.className = "status-dot recording";
    systemText.textContent = "Monitoring Active";
    btnStop.disabled = false;

    initCameraLoop();

    // Reset charts
    attentionChart.data.labels = [];
    attentionChart.data.datasets[0].data = [];
    attentionChart.data.datasets[1].data = [];
    attentionChart.update();

    communicationChart.data.labels = [];
    communicationChart.data.datasets[0].data = [];
    communicationChart.data.datasets[1].data = [];
    communicationChart.update();

    // Start Chart logging intervals
    let seconds = 0;
    chartInterval = setInterval(() => {
        seconds++;
        const timeStr = `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;

        chartHistoryTime.push(timeStr);
        chartHistoryAttention.push(currentRealtimeState.attention);
        chartHistoryFocus.push(currentRealtimeState.focus);
        chartHistoryConfidence.push(currentRealtimeState.confidence);
        chartHistoryStress.push(currentRealtimeState.stress);

        attentionChart.data.labels.push(timeStr);
        attentionChart.data.datasets[0].data.push(currentRealtimeState.attention);
        attentionChart.data.datasets[1].data.push(currentRealtimeState.focus);
        if (attentionChart.data.labels.length > 30) {
            attentionChart.data.labels.shift();
            attentionChart.data.datasets[0].data.shift();
            attentionChart.data.datasets[1].data.shift();
        }
        attentionChart.update();

        communicationChart.data.labels.push(timeStr);
        communicationChart.data.datasets[0].data.push(currentRealtimeState.confidence);
        communicationChart.data.datasets[1].data.push(currentRealtimeState.stress);
        if (communicationChart.data.labels.length > 30) {
            communicationChart.data.labels.shift();
            communicationChart.data.datasets[0].data.shift();
            communicationChart.data.datasets[1].data.shift();
        }
        communicationChart.update();
    }, 1000);

    addConsoleLog("00:00", "System", "Monitoring Started", "Candidate face tracking enabled.");
}

// Stop Monitoring
function stopMonitoring() {
    if (!monitoringActive) return;
    monitoringActive = false;

    btnStop.disabled = true;
    if (chartInterval) {
        clearInterval(chartInterval);
        chartInterval = null;
    }

    // Stop streams
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (mediaPipeCamera) {
        mediaPipeCamera.stop();
        mediaPipeCamera = null;
    }

    systemDot.className = "status-dot";
    systemText.textContent = "Session Ended";
    btnStart.disabled = false;
    btnReset.style.display = "block";
    roiOverlay.style.display = "none";
    faceMeshStatusBadge.style.display = "flex";

    // Compute final violations
    const sessionDur = (Date.now() - (sessionStartTime || Date.now())) / 1000;
    scoringEngine.explanationEngine.endActiveViolations(sessionDur);

    // Render report
    compileReport();
}

// Reset Dashboard
function resetDashboard() {
    document.getElementById("dashboard-view").style.display = "flex";
    document.getElementById("report-view").style.display = "none";
    btnReset.style.display = "none";
    systemText.textContent = "System Idle";
    systemDot.className = "status-dot";
    
    // Clear gauges
    updateProgressRing(gaugeScore, valScore, 0);
    updateProgressRing(gaugeAttention, valAttention, 0);
    updateProgressRing(gaugeCommunication, valCommunication, 0);
    updateProgressRing(gaugeIntegrity, valIntegrity, 100);
    updateProgressRing(gaugeStress, valStress, 0);
    updateProgressRing(gaugeInferenceConf, valInferenceConf, 100);

    // Clear traceback
    document.getElementById("trace-attention").textContent = "+0";
    document.getElementById("trace-focus").textContent = "+0";
    document.getElementById("trace-comm").textContent = "+0";
    document.getElementById("trace-gaze").textContent = "-0";
    document.getElementById("trace-stress").textContent = "-0";
    document.getElementById("trace-integrity").textContent = "-0";
    
    // Clear console logs
    logConsole.innerHTML = '<div class="log-empty" id="log-empty-state">No events logged yet. Start monitoring.</div>';
    logCounter.textContent = "0 events";
    
    cropROI = { x: 0, y: 0, w: 0, h: 0 };
    selectedBox = null;
}

// Audio Processing Setup (Microphone volume tracking)
function startAudioProcessing(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    function checkAudio() {
        if (!monitoringActive) return;
        analyser.getFloatTimeDomainData(dataArray);
        
        let sumSquares = 0.0;
        for (let i = 0; i < bufferLength; i++) {
            sumSquares += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sumSquares / bufferLength);
        currentRealtimeState.speech_energy = Math.min(1.0, rms * 15.0);
        currentRawStats.speaking = rms > 0.015; // original threshold
        
        requestAnimationFrame(checkAudio);
    }
    checkAudio();
}

// Process Frame with MediaPipe Mesh
function onMeshResults(results) {
    if (!monitoringActive) return;

    try {
        const faceCount = results.multiFaceLandmarks ? results.multiFaceLandmarks.length : 0;
        const landmarks = faceCount > 0 ? results.multiFaceLandmarks[0] : null;

        const w = canvasEl.width;
        const h = canvasEl.height;

        // Environmental Checks
        currentEnvMetrics.face_visible = faceCount > 0;
        
        // Lighting brightness check
        const imgData = ctx.getImageData(0, 0, w, h);
        const pixels = imgData.data;
        let brightnessSum = 0;
        let brightnessCount = 0;
        for (let i = 0; i < pixels.length; i += 40) { // sample pixels
            brightnessSum += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
            brightnessCount++;
        }
        const avgBrightness = brightnessSum / brightnessCount;
        currentEnvMetrics.lighting_ok = avgBrightness >= 55.0;

        // Blur check using 64x64 downsampled Laplacian variance filter
        const laplacianVar = calculateLaplacianVariance();
        currentEnvMetrics.blur_ok = (laplacianVar * 5) >= 90.0;

        // Jitter & ROI Stability checks
        if (currentEnvMetrics.face_visible && landmarks && landmarks[1] && landmarks[234] && landmarks[454]) {
            // Draw mesh contour on top of current canvas contents
            drawMeshContours(landmarks, w, h);

            // Landmark stability (jitter check on nose tip landmark index 1)
            const noseLm = landmarks[1];
            const nosePos = { x: noseLm.x * w, y: noseLm.y * h };
            if (prevNosePos !== null) {
                const dist = Math.hypot(nosePos.x - prevNosePos.x, nosePos.y - prevNosePos.y);
                jitterHistory.push(dist);
                if (jitterHistory.length > 15) jitterHistory.shift();
            }
            prevNosePos = nosePos;

            if (jitterHistory.length >= 5) {
                const avgJitter = jitterHistory.reduce((acc, v) => acc + v, 0) / jitterHistory.length;
                currentEnvMetrics.landmarks_stable = avgJitter <= 6.0;
            } else {
                currentEnvMetrics.landmarks_stable = true;
            }

            // ROI stability (width variance check using width landmarks 234 and 454)
            const p1 = landmarks[234];
            const p2 = landmarks[454];
            const faceW = Math.hypot((p2.x - p1.x) * w, (p2.y - p1.y) * h);
            faceWidthHistory.push(faceW);
            if (faceWidthHistory.length > 20) faceWidthHistory.shift();

            if (faceWidthHistory.length >= 10) {
                const meanW = faceWidthHistory.reduce((acc, v) => acc + v, 0) / faceWidthHistory.length;
                if (meanW > 0) {
                    const sqDiffSum = faceWidthHistory.reduce((acc, v) => acc + (v - meanW) ** 2, 0);
                    const stdDev = Math.sqrt(sqDiffSum / faceWidthHistory.length);
                    currentEnvMetrics.roi_stable = (stdDev / meanW) <= 0.08;
                } else {
                    currentEnvMetrics.roi_stable = true;
                }
            } else {
                currentEnvMetrics.roi_stable = true;
            }
        } else {
            prevNosePos = null;
            currentEnvMetrics.landmarks_stable = true;
            currentEnvMetrics.roi_stable = true;
        }

        // Process raw stats
        const integrityFlags = integrityMonitor.process(landmarks, faceCount);
        currentRawStats.integrity_flags = integrityFlags;

        if (currentEnvMetrics.face_visible && landmarks) {
            const blinkCount = blinkAnalyzer.process(landmarks, w, h);
            const gazeDir = gazeTracker.process(landmarks, w, h);
            const headPose = headPoseEstimator.process(landmarks);

            // Core realtime calculations
            const emotionObj = emotionInference.process(
                blinkCount,
                gazeDir,
                headPose,
                currentRawStats.speaking,
                landmarks,
                w,
                h
            );

            currentRealtimeState = emotionObj;
            
            currentRawStats.blinks = blinkCount;
            currentRawStats.gaze = gazeDir;
            currentRawStats.head_pose = headPose;
        } else {
            // Fallbacks for face absent
            currentRealtimeState.feeling = "Neutral";
            currentRealtimeState.attention = 0.2;
            currentRealtimeState.focus = 0.0;
            currentRealtimeState.stress = Math.min(1.0, currentRealtimeState.stress + 0.02);
            currentRealtimeState.confidence = Math.max(0.0, currentRealtimeState.confidence - 0.02);

            currentRawStats.blinks = blinkAnalyzer ? blinkAnalyzer.blinkCount : 0;
            currentRawStats.gaze = "Unknown";
            currentRawStats.head_pose = "Unknown";
        }

        // scoring calculations
        const scores = scoringEngine.calculate(
            currentRealtimeState,
            integrityFlags,
            currentEnvMetrics,
            currentRawStats
        );

        // timeline analytics checks
        timelineAnalytics.process(
            currentRealtimeState,
            scores,
            integrityFlags,
            currentEnvMetrics,
            currentRawStats
        );

        // Render indicators to Dashboard UI
        updateDashboardUI(scores);
    } catch (err) {
        console.error("Error in onMeshResults core thread: ", err);
    }
}

// Compute Laplacian Variance for blur detection
function calculateLaplacianVariance() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 64;
    tempCanvas.height = 64;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvasEl, 0, 0, 64, 64);
    
    const imgData = tempCtx.getImageData(0, 0, 64, 64);
    const data = imgData.data;
    const gray = new Float32Array(64 * 64);
    
    // Grayscale
    for (let i = 0; i < 4096; i++) {
        gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    }
    
    // Laplacian kernel
    const laplacian = new Float32Array(64 * 64);
    let sum = 0;
    let count = 0;
    
    for (let y = 1; y < 63; y++) {
        for (let x = 1; x < 63; x++) {
            const idx = y * 64 + x;
            const val = gray[idx - 64] + gray[idx - 1] - 4 * gray[idx] + gray[idx + 1] + gray[idx + 64];
            laplacian[idx] = val;
            sum += val;
            count++;
        }
    }
    
    const mean = sum / count;
    let sumSqDiff = 0;
    
    for (let y = 1; y < 63; y++) {
        for (let x = 1; x < 63; x++) {
            const idx = y * 64 + x;
            sumSqDiff += (laplacian[idx] - mean) ** 2;
        }
    }
    
    return sumSqDiff / count;
}

// Draw contours on Canvas
function drawMeshContours(landmarks, w, h) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(59, 130, 246, 0.5)"; // primary glow color
    ctx.fillStyle = "rgba(59, 130, 246, 0.8)";

    const drawPath = (indices, close = false) => {
        if (!indices || indices.length === 0) return;
        const start = landmarks[indices[0]];
        if (!start) return;
        ctx.beginPath();
        ctx.moveTo(start.x * w, start.y * h);
        for (let i = 1; i < indices.length; i++) {
            const pt = landmarks[indices[i]];
            if (pt) {
                ctx.lineTo(pt.x * w, pt.y * h);
            }
        }
        if (close) ctx.closePath();
        ctx.stroke();
    };

    // Draw Eye Contours
    drawPath([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], true);
    drawPath([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], true);

    // Draw Eyebrows
    drawPath([70, 63, 105, 66, 107]);
    drawPath([300, 293, 334, 296, 336]);

    // Draw Lips
    drawPath([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95, 88], true);

    // Draw Irises
    ctx.fillStyle = "#10b981"; // emerald iris indicators
    const drawIris = (idx) => {
        const pt = landmarks[idx];
        if (!pt) return;
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, 3, 0, 2 * Math.PI);
        ctx.fill();
    };
    drawIris(468); // Left iris center
    drawIris(473); // Right iris center
    
    // Draw Face Oval Outline
    drawPath([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], true);
}

// Update SVG progress ring circular stroke dash offset
function updateProgressRing(circleElement, textElement, value) {
    const val = Math.round(value);
    const radius = circleElement.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    
    circleElement.style.strokeDasharray = `${circumference} ${circumference}`;
    const offset = circumference - (val / 100) * circumference;
    circleElement.style.strokeDashoffset = offset;
    
    textElement.textContent = `${val}%`;
}

// Update dashboard panel scores, traceback metrics, and quality badges
function updateDashboardUI(scores) {
    // Progress Gauges
    updateProgressRing(gaugeScore, valScore, scores.session_score);
    updateProgressRing(gaugeAttention, valAttention, scores.session_attention);
    updateProgressRing(gaugeCommunication, valCommunication, scores.session_confidence);
    updateProgressRing(gaugeIntegrity, valIntegrity, scores.integrity_score);
    updateProgressRing(gaugeStress, valStress, currentRealtimeState.stress * 100);
    updateProgressRing(gaugeInferenceConf, valInferenceConf, scores.inference_confidence * 100);

    // Color code inference gauge
    const conf = scores.inference_confidence;
    gaugeInferenceConf.setAttribute("stroke", conf > 0.8 ? "#10b981" : (conf > 0.5 ? "#f59e0b" : "#ef4444"));

    // Quality badges
    updateQualityBadge(qualityLighting, currentEnvMetrics.lighting_ok, "OK", "Low");
    updateQualityBadge(qualityBlur, currentEnvMetrics.blur_ok, "Clear", "Blurry");
    updateQualityBadge(qualityJitter, currentEnvMetrics.landmarks_stable, "Stable", "Jitter");
    updateQualityBadge(qualityRoi, currentEnvMetrics.roi_stable, "Stable", "Unstable");

    // Traceback scores list
    updateTracebackRow(traceAttention, scores.contributors["Attention Consistency"]);
    updateTracebackRow(traceFocus, scores.contributors["Stable Head Pose"]);
    updateTracebackRow(traceComm, scores.contributors["Strong Communication"]);
    updateTracebackRow(traceGaze, scores.contributors["Frequent Gaze Deviation"]);
    updateTracebackRow(traceStress, scores.contributors["Elevated Stress"]);
    updateTracebackRow(traceIntegrity, scores.contributors["Integrity Penalty"]);
}

function updateQualityBadge(element, isOk, okText, badText) {
    if (isOk) {
        element.className = "quality-badge badge-success";
        element.textContent = okText;
    } else {
        element.className = "quality-badge badge-danger";
        element.textContent = badText;
    }
}

function updateTracebackRow(element, val) {
    const sign = val >= 0 ? "+" : "";
    element.textContent = `${sign}${val}`;
    element.className = `traceback-value ${val >= 0 ? 'positive' : 'negative'}`;
}

// Render dynamic Toast alerts (slide in warning cards)
function showToast(type, name, details) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    
    toast.className = `toast toast-${type.toLowerCase()}`;
    
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-title" style="display:flex; justify-content:space-between; align-items:center;">
                <span>${name}</span>
                <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">${timeStr}</span>
            </div>
            <div class="toast-body">${details}</div>
        </div>
    `;

    container.appendChild(toast);

    // Auto delete after 5.5s
    setTimeout(() => {
        toast.style.animation = "slideIn 0.3s ease reverse forwards";
        setTimeout(() => {
            if (toast.parentNode) container.removeChild(toast);
        }, 300);
    }, 5500);
}

// Write to timeline console logs list
function addConsoleLog(timeStr, type, name, details) {
    logEmptyState.style.display = "none";
    
    const entry = document.createElement("div");
    entry.className = "log-entry";
    
    entry.innerHTML = `
        <span class="log-time">[${timeStr}]</span>
        <span class="log-type-${type.toLowerCase()}">${type.toUpperCase()}: ${name}</span>
        <span class="log-details"> - ${details}</span>
    `;

    logConsole.appendChild(entry);
    logConsole.scrollTop = logConsole.scrollHeight;

    const count = logConsole.getElementsByClassName("log-entry").length;
    logCounter.textContent = `${count} event${count > 1 ? 's' : ''}`;
}

// Generate Post-Session Recruiter Evaluation Report
function compileReport() {
    // Hide Dashboard and show Report Screen
    document.getElementById("dashboard-view").style.display = "none";
    const reportWorkspace = document.getElementById("report-view");
    reportWorkspace.style.display = "flex";

    // Set Timestamps
    const reportTimestamp = document.getElementById("report-timestamp");
    reportTimestamp.textContent = `Generated on: ${new Date().toLocaleString()}`;

    const finalAverages = scoringEngine.sessionAverages;
    const explanationEngine = scoringEngine.explanationEngine;
    
    if (finalAverages.attention.length === 0) {
        alert("No behavioral data was captured during this session.");
        return;
    }

    const sessionScore = scoringEngine.calculate(
        currentRealtimeState,
        currentRawStats.integrity_flags,
        currentEnvMetrics,
        currentRawStats
    );

    // Compute average scores
    const avgScore = sessionScore.session_score;
    const avgIntegrity = sessionScore.integrity_score;
    const avgConfidence = finalAverages.inference_confidence.reduce((acc, v) => acc + v, 0) / finalAverages.inference_confidence.length;

    // Fill Executive metrics cards
    document.getElementById("report-val-score").textContent = `${avgScore}/100`;
    document.getElementById("report-val-integrity").textContent = `${avgIntegrity.toFixed(1)}%`;
    document.getElementById("report-val-confidence").textContent = `${Math.round(avgConfidence * 100)}%`;

    // Fill Detailed traceback contributions bar charts
    const tracebackBars = document.getElementById("report-traceback-bars");
    tracebackBars.innerHTML = "";
    
    for (const [factor, val] of Object.entries(sessionScore.contributors)) {
        const isPositive = val >= 0;
        const sign = isPositive ? "+" : "";
        const maxLimit = factor.includes("Attention") || factor.includes("Communication") ? 40 : (factor.includes("Head") || factor.includes("Stress") ? 20 : 10);
        const percent = Math.min(100, Math.max(0, (Math.abs(val) / maxLimit) * 100));

        const barRow = document.createElement("div");
        barRow.className = "bar-row";
        barRow.innerHTML = `
            <div class="bar-label-container">
                <span>${factor}</span>
                <span class="traceback-value ${isPositive ? 'positive' : 'negative'}">${sign}${val}</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill ${isPositive ? 'positive' : 'negative'}" style="width: ${percent}%"></div>
            </div>
        `;
        tracebackBars.appendChild(barRow);
    }

    // Fill Recruiter Insights Lists
    const insightsList = document.getElementById("report-insights-list");
    insightsList.innerHTML = "";
    const insights = explanationEngine.generateRecruiterSummary(avgScore, avgIntegrity);
    insights.forEach(ins => {
        const li = document.createElement("li");
        li.className = "bullet-item";
        li.innerHTML = `<span class="bullet-marker">•</span><span class="bullet-text">${ins}</span>`;
        insightsList.appendChild(li);
    });

    // Fill Metric explanations panels
    fillExplanationList("report-attention-exp", sessionScore.attention_explanation.explanation);
    fillExplanationList("report-confidence-exp", sessionScore.confidence_explanation.explanation);
    fillExplanationList("report-integrity-exp", sessionScore.integrity_explanation.explanation);

    // Fill Timeline event table
    const tableBody = document.getElementById("report-events-table");
    tableBody.innerHTML = "";
    
    if (timelineAnalytics.events.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; color: var(--text-muted); font-style: italic;">No events logged during the session.</td>
            </tr>
        `;
    } else {
        timelineAnalytics.events.forEach(e => {
            const tr = document.createElement("tr");
            const typeBadge = e.type === "Integrity" ? "badge-danger" : "badge-success";
            
            tr.innerHTML = `
                <td style="font-family: var(--font-mono);">${e.time_str}</td>
                <td><span class="badge-event ${typeBadge}">${e.name}</span></td>
                <td style="font-family: var(--font-mono);">${Math.round(e.reason_data.confidence * 100)}%</td>
                <td>${e.reason_data.reason.join(", ")}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    // Populate report editor code textarea
    const reportCodeEditor = document.getElementById("report-code-editor");
    if (reportCodeEditor) {
        reportCodeEditor.value = generateMarkdownReport();
    }
    
    // No Git status badges to reset

    // Reset UX feedback inputs
    selectedUXRating = 0;
    const stars = document.querySelectorAll("#ux-star-rating .star");
    stars.forEach(s => s.classList.remove("selected"));
    const uxComments = document.getElementById("ux-comments");
    if (uxComments) uxComments.value = "";
}

function fillExplanationList(elementId, items) {
    const container = document.getElementById(elementId);
    container.innerHTML = "";
    items.forEach(item => {
        const li = document.createElement("li");
        li.className = "bullet-item";
        li.innerHTML = `<span class="bullet-marker">•</span><span class="bullet-text">${item}</span>`;
        container.appendChild(li);
    });
}

// Copy Markdown Session Report to Clipboard
// Copy Markdown Session Report to Clipboard
function copyMarkdownReport() {
    const reportCodeEditor = document.getElementById("report-code-editor");
    const md = reportCodeEditor ? reportCodeEditor.value : generateMarkdownReport();

    navigator.clipboard.writeText(md).then(() => {
        alert("Markdown report successfully copied to clipboard!");
    }).catch(err => {
        console.error("Could not copy markdown report: ", err);
        alert("Failed to copy report. Markdown contents logged to developer console.");
        console.log(md);
    });
}

// Generate the Raw Markdown String
function generateMarkdownReport() {
    const finalAverages = scoringEngine.sessionAverages;
    const explanationEngine = scoringEngine.explanationEngine;
    
    const sessionScore = scoringEngine.calculate(
        currentRealtimeState,
        currentRawStats.integrity_flags,
        currentEnvMetrics,
        currentRawStats
    );

    const avgScore = sessionScore.session_score;
    const avgIntegrity = sessionScore.integrity_score;
    const avgConfidence = finalAverages.inference_confidence.reduce((acc, v) => acc + v, 0) / finalAverages.inference_confidence.length;

    let md = `# AI Interview Session Explainability Report\n\n`;
    md += `Generated on: ${new Date().toLocaleString()}\n\n`;
    md += `## Executive Summary\n`;
    md += `- **Final Session Score**: ${avgScore}/100\n`;
    md += `- **Overall Integrity**: ${avgIntegrity.toFixed(1)}%\n`;
    md += `- **Average Inference Confidence**: ${Math.round(avgConfidence * 100)}%\n\n`;

    md += `## Scoring Traceback Breakdown\n`;
    md += `Every final evaluation score is traceable to its constituent behavioral variables. The contributors sum up to the final score:\n\n`;
    md += `| Contributor Factor | Impact on Score |\n`;
    md += `| :--- | :--- |\n`;
    for (const [factor, val] of Object.entries(sessionScore.contributors)) {
        const sign = val >= 0 ? "+" : "";
        md += `| ${factor} | ${sign}${val} |\n`;
    }
    md += `| **Final Session Score** | **${avgScore}** |\n\n`;

    md += `## Structured Metric Explanations\n\n`;
    
    md += `### Attention Score\n`;
    md += `**Evidence & Observations**:\n`;
    sessionScore.attention_explanation.explanation.forEach(line => {
        md += `- ${line}\n`;
    });
    md += `\n`;

    md += `### Confidence Score\n`;
    md += `**Evidence & Observations**:\n`;
    sessionScore.confidence_explanation.explanation.forEach(line => {
        md += `- ${line}\n`;
    });
    md += `\n`;

    md += `### Integrity Evaluation\n`;
    md += `**Evidence & Observations**:\n`;
    sessionScore.integrity_explanation.explanation.forEach(line => {
        md += `- ${line}\n`;
    });
    md += `\n`;

    md += `## Behavioral & Integrity Timeline Events\n`;
    if (timelineAnalytics.events.length === 0) {
        md += `No major behavioral anomalies or warnings were logged during the session.\n\n`;
    } else {
        md += `| Timestamp | Event Name | Confidence | Contributing Causes |\n`;
        md += `| :--- | :--- | :--- | :--- |\n`;
        timelineAnalytics.events.forEach(e => {
            md += `| ${e.time_str} | ${e.name} | ${e.reason_data.confidence} | ${e.reason_data.reason.join(", ")} |\n`;
        });
        md += `\n`;
    }

    md += `## Recruiter Insights & Recommendations\n`;
    const insights = explanationEngine.generateRecruiterSummary(avgScore, avgIntegrity);
    insights.forEach(insight => {
        md += `- ${insight}\n`;
    });

    return md;
}

// Setup Event Listeners for Report Finalization and UX Feedback
function setupReportFinalizeListeners() {
    const btnSaveReport = document.getElementById("btn-save-report");
    const finalizeConfirmModal = document.getElementById("finalize-confirm-modal");
    const btnFinalizeConfirmYes = document.getElementById("btn-finalize-confirm-yes");
    const btnFinalizeConfirmNo = document.getElementById("btn-finalize-confirm-no");
    const reportCodeEditor = document.getElementById("report-code-editor");

    // UX Feedback setup
    setupUXFeedbackHandlers();

    if (btnSaveReport) {
        btnSaveReport.addEventListener("click", () => {
            if (!reportCodeEditor.value.trim()) {
                alert("The report markdown code is empty.");
                return;
            }
            finalizeConfirmModal.style.display = "flex";
        });
    }

    if (btnFinalizeConfirmNo) {
        btnFinalizeConfirmNo.addEventListener("click", () => {
            finalizeConfirmModal.style.display = "none";
        });
    }

    if (btnFinalizeConfirmYes) {
        btnFinalizeConfirmYes.addEventListener("click", () => {
            finalizeConfirmModal.style.display = "none";
            
            showToast("Behavior", "Report Save", "Saving final report contents...");
            
            setTimeout(() => {
                // Copy final content to clipboard for easy pasting/saving
                navigator.clipboard.writeText(reportCodeEditor.value).then(() => {
                    showToast("Integrity", "Save Success", "Report finalized and copied to clipboard!");
                }).catch(err => {
                    console.error("Could not copy finalized report: ", err);
                    showToast("Behavior", "Save Info", "Report finalized successfully!");
                });
            }, 500);
        });
    }
}

let selectedUXRating = 0;

function setupUXFeedbackHandlers() {
    const stars = document.querySelectorAll("#ux-star-rating .star");
    const uxComments = document.getElementById("ux-comments");

    stars.forEach(star => {
        star.addEventListener("click", (e) => {
            const val = parseInt(e.target.getAttribute("data-value"));
            selectedUXRating = val;
            
            stars.forEach((s, idx) => {
                if (idx < val) {
                    s.classList.add("selected");
                } else {
                    s.classList.remove("selected");
                }
            });

            updateReportCodeWithFeedback();
        });
    });

    if (uxComments) {
        uxComments.addEventListener("input", () => {
            updateReportCodeWithFeedback();
        });
    }
}

function updateReportCodeWithFeedback() {
    const reportCodeEditor = document.getElementById("report-code-editor");
    if (!reportCodeEditor) return;

    let baseMd = generateMarkdownReport();
    const comments = document.getElementById("ux-comments").value.trim() || "None";

    if (selectedUXRating > 0 || comments !== "None") {
        let feedbackMd = `\n## User Experience & Product Review\n`;
        feedbackMd += `- **Platform Rating**: ${"★".repeat(selectedUXRating)}${"☆".repeat(5 - selectedUXRating)} (${selectedUXRating}/5)\n`;
        feedbackMd += `- **User Comments**: ${comments}\n`;
        baseMd += feedbackMd;
    }

    reportCodeEditor.value = baseMd;
}
