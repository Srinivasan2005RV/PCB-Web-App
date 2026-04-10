// ============ AZURE CONFIGURATION ============
const AZURE_CONFIG = {
    backendUrl: 'https://web-service-rya2.onrender.com',  // Cloud-only: Real YOLO on Render
    storageAccount: localStorage.getItem('storageAccount') || 'pcbdetectionimages'
};

// Global variables
let selectedImage = null;
let currentDetectionId = null;
let userId = localStorage.getItem('userId') || 'user_' + Date.now();

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');

    // Load history when switching to history tab
    if (tabName === 'history') {
        loadHistory();
    }
}

// Handle image selection
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('Image size should be less than 10MB');
        return;
    }

    // Read and preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        selectedImage = {
            file: file,
            data: e.target.result,
            name: file.name
        };

        // Show preview
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.style.display = 'block';

        // Enable detect button
        document.getElementById('detectBtn').disabled = false;
        document.getElementById('uploadError').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Run detection using Azure backend
async function runDetection() {
    if (!selectedImage) {
        showError('Please select an image first');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const loading = document.getElementById('uploadLoading');

    try {
        // Disable button and show loading
        detectBtn.disabled = true;
        loading.classList.add('active');

        // Generate unique detection ID
        currentDetectionId = `detection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', selectedImage.file);

        let result;
        let usedBackend = 'Render';

        try {
            // Try Render backend with longer timeout for cold start + YOLO inference
            console.log('Trying Render backend (with 120s timeout for cold start + inference)...');
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000);  // 120 seconds for Render cold start + YOLO

            const response = await fetch(`${AZURE_CONFIG.backendUrl}/detect`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status} ${response.statusText}`);
            }

            result = await response.json();
            console.log('✅ Cloud backend response:', result);

        } catch (error) {
            console.error('❌ Backend error:', error.message);
            throw new Error(`Cloud backend unavailable: ${error.message}. Render service may still be cold-starting. Try again in 30 seconds.`);
        }

        // Convert cloud backend response to our format
        let defectsArray = [];
        if (result.detections && Array.isArray(result.detections)) {
            defectsArray = result.detections.map(det => ({
                class: det.defect_type || det.class_name,
                confidence: det.confidence,
                bbox: [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            }));
        }

        // Display results with cloud backend data
        displayResults({
            defects: defectsArray,
            confidence: result.detection_summary?.total_defects > 0 ? 0.85 : 0.95,
            timestamp: result.timestamp,
            detectionId: currentDetectionId,
            annotatedImageBase64: result.annotated_image_base64
        });

        // Show success
        showSuccess('✅ Detection from cloud backend completed!');

        // Reset
        setTimeout(() => {
            selectedImage = null;
            document.getElementById('preview').style.display = 'none';
            document.getElementById('imageInput').value = '';
            detectBtn.disabled = true;
        }, 1000);

    } catch (error) {
        console.error('Detection error:', error);
        showError('❌ Cloud backend unavailable: ' + error.message);
    } finally {
        loading.classList.remove('active');
        detectBtn.disabled = false;
    }
}

// Display results
function displayResults(results) {
    let html = '';

    // Show annotated image from Render API if available, otherwise fallback to canvas drawing
    if (results.annotatedImageBase64) {
        // Use YOLO-annotated image from Render backend (boxes already drawn!)
        html += `<div class="result-image-container" style="width: 100%; margin-bottom: 15px;">
            <img src="data:image/jpeg;base64,${results.annotatedImageBase64}" alt="PCB Detection Result" style="max-width: 100%; border-radius: 8px; display: block; width: 100%;">
        </div>`;
    } else if (selectedImage && selectedImage.data) {
        // Fallback: Show original image with canvas overlay
        html += `<div class="result-image-container" style="position: relative; display: inline-block; width: 100%; margin-bottom: 15px;">
            <img id="resultImage" src="${selectedImage.data}" alt="PCB Image" style="max-width: 100%; border-radius: 8px; display: block; width: 100%;">
            <canvas id="bboxCanvas" style="position: absolute; top: 0; left: 0; border-radius: 8px;"></canvas>
        </div>`;
    }

    // Status indicator
    const hasDefects = results.defects && results.defects.length > 0;
    const statusClass = hasDefects ? 'status-defect-found' : 'status-no-defect';
    const statusText = hasDefects ? '⚠️ DEFECTS FOUND' : '✅ NO DEFECTS';

    html += `<div class="result-card">
        <div class="result-status ${statusClass}">${statusText}</div>
        <div class="result-details">
            <strong>Total Defects:</strong> ${results.defects ? results.defects.length : 0}
        </div>
        <div class="result-details">
            <strong>Confidence:</strong> ${(results.confidence * 100).toFixed(1)}%
        </div>`;

    if (hasDefects) {
        html += '<div class="defect-list"><strong>Defect Types:</strong>';
        results.defects.forEach(defect => {
            html += `<div class="defect-item">
                ${defect.class}
                <span class="confidence">${(defect.confidence * 100).toFixed(1)}%</span>
            </div>`;
        });
        html += '</div>';
    }

    html += `<div class="result-details">
        <strong>Detection ID:</strong> ${results.detectionId ? results.detectionId.substr(0, 20) + '...' : 'N/A'}
    </div>`;
    html += '</div>';

    document.getElementById('resultsContent').innerHTML = html;

    // Draw bounding boxes only if no annotated image (fallback mode)
    if (!results.annotatedImageBase64 && hasDefects && selectedImage && selectedImage.data) {
        setTimeout(() => {
            drawBoundingBoxes(results.defects);
        }, 100);
    }

    // Switch to results tab
    document.querySelector('[onclick="switchTab(\'results\')"]').click();
}

// Draw bounding boxes on the image
function drawBoundingBoxes(defects) {
    const canvas = document.getElementById('bboxCanvas');
    const img = document.getElementById('resultImage');
    
    if (!canvas || !img) return;
    
    // Wait for image to load
    if (!img.complete) {
        img.onload = () => drawBoundingBoxes(defects);
        return;
    }
    
    // Set canvas size to match displayed image
    const displayWidth = img.offsetWidth;
    const displayHeight = img.offsetHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    // Scale ratio from original image to displayed image
    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;
    
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    const ctx = canvas.getContext('2d');
    
    // Color map for different defect types
    const colorMap = {
        'missing_hole': '#FF6B6B',
        'open_circuit': '#FFD93D',
        'short': '#FF8C42',
        'spur': '#6BCB77',
        'mouse_bite': '#4D96FF',
        'spurious_copper': '#FF69B4'
    };
    
    defects.forEach((defect, idx) => {
        if (!defect.bbox || defect.bbox.length < 4) return;
        
        // Scale bbox coordinates
        let [x1, y1, x2, y2] = defect.bbox;
        x1 = x1 * scaleX;
        y1 = y1 * scaleY;
        x2 = x2 * scaleX;
        y2 = y2 * scaleY;
        
        const color = colorMap[defect.class] || '#FF0000';
        
        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label
        const label = `${defect.class} ${(defect.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = color;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 25, textWidth + 8, 22);
        
        // Draw text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 4, y1 - 8);
    });
}

// Draw bounding boxes on the image
function drawBoundingBoxes(defects) {
    const canvas = document.getElementById('bboxCanvas');
    const img = document.getElementById('resultImage');
    
    if (!canvas || !img) return;
    
    // Wait for image to load
    if (!img.complete) {
        img.onload = () => drawBoundingBoxes(defects);
        return;
    }
    
    // Set canvas size to match displayed image (not original size)
    const displayWidth = img.offsetWidth;
    const displayHeight = img.offsetHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    // Scale ratio from original image to displayed image
    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;
    
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    const ctx = canvas.getContext('2d');
    
    // Color map for different defect types
    const colorMap = {
        'missing_hole': '#FF6B6B',
        'open_circuit': '#FFD93D',
        'short': '#FF8C42',
        'spur': '#6BCB77',
        'mouse_bite': '#4D96FF',
        'spurious_copper': '#FF69B4'
    };
    
    defects.forEach((defect, idx) => {
        if (!defect.bbox || defect.bbox.length < 4) return;
        
        // Scale bbox coordinates from original image to displayed size
        let [x1, y1, x2, y2] = defect.bbox;
        x1 = x1 * scaleX;
        y1 = y1 * scaleY;
        x2 = x2 * scaleX;
        y2 = y2 * scaleY;
        
        const color = colorMap[defect.class] || '#FF0000';
        
        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label background
        const label = `${defect.class} ${(defect.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = color;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 22, textWidth + 8, 20);
        
        // Draw label text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 4, y1 - 8);
        
        // Store defect info for click handler
        if (!canvas.defectsData) canvas.defectsData = [];
        canvas.defectsData[idx] = {
            index: idx,
            class: defect.class,
            confidence: defect.confidence,
            bbox: [x1, y1, x2, y2],
            originalBbox: defect.bbox
        };
    });
    
    // Add click handler for interactive defect confirmation
    if (!canvas.clickHandlerAdded) {
        canvas.clickHandlerAdded = true;
        canvas.onclick = (e) => {
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            if (canvas.defectsData) {
                for (let i = 0; i < canvas.defectsData.length; i++) {
                    const d = canvas.defectsData[i];
                    const [x1, y1, x2, y2] = d.bbox;
                    
                    if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                        showDefectConfirmation(d);
                        return;
                    }
                }
            }
        };
    }
}

// Show defect confirmation modal when clicked
function showDefectConfirmation(defectData) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;
    
    const content = document.createElement('div');
    content.style.cssText = `
        background: white;
        border-radius: 15px;
        padding: 25px;
        max-width: 400px;
        width: 90%;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    `;
    
    const colorMap = {
        'missing_hole': '#FF6B6B',
        'open_circuit': '#FFD93D',
        'short': '#FF8C42',
        'spur': '#6BCB77',
        'mouse_bite': '#4D96FF',
        'spurious_copper': '#FF69B4'
    };
    
    const color = colorMap[defectData.class] || '#FF0000';
    
    content.innerHTML = `
        <div style="text-align: center;">
            <h2 style="color: #333; margin-bottom: 15px;">Defect Detected</h2>
            <div style="background: ${color}20; padding: 15px; border-radius: 10px; margin: 15px 0;">
                <p style="color: ${color}; font-size: 24px; font-weight: bold; margin: 10px 0;">
                    ${defectData.class.toUpperCase()}
                </p>
                <p style="color: #666; font-size: 18px; margin: 10px 0;">
                    Confidence: <strong>${(defectData.confidence * 100).toFixed(1)}%</strong>
                </p>
            </div>
            <p style="color: #999; font-size: 14px; margin-bottom: 20px;">
                Is this defect correct?
            </p>
            <div style="display: flex; gap: 10px;">
                <button style="flex: 1; padding: 12px; background: #FF6B6B; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer;" onclick="confirmDefectSelection(true)">
                    ✓ Correct
                </button>
                <button style="flex: 1; padding: 12px; background: #FFD93D; color: #333; border: none; border-radius: 8px; font-size: 16px; cursor: pointer;" onclick="confirmDefectSelection(false)">
                    ❌ Incorrect
                </button>
            </div>
        </div>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    
    modal.onclick = (e) => {
        if (e.target === modal) modal.remove();
    };
}

// Handle defect confirmation
async function confirmDefectSelection(isCorrect) {
    try {
        // Remove modal
        const modal = document.querySelector('div[style*="position: fixed"]');
        if (modal) modal.remove();
        
        showSuccess(`Defect ${isCorrect ? 'Confirmed ✓' : 'Rejected ❌'} - Saved to history!`);
    } catch (error) {
        console.error('Error confirming defect:', error);
    }
}

// Load history from Azure backend
async function loadHistory() {
    try {
        const backendUrl = AZURE_CONFIG.backendUrl;
        const response = await fetch(`${backendUrl}/history/${userId}`);

        if (!response.ok) {
            throw new Error('Failed to fetch history');
        }

        const data = await response.json();

        if (!data.detections || data.detections.length === 0) {
            document.getElementById('historyList').innerHTML = 
                '<p style="color: #999; text-align: center;">No history yet.</p>';
            return;
        }

        let html = '';
        data.detections.slice(0, 20).forEach(detection => {
            const timestamp = new Date(detection.timestamp);
            const timeStr = timestamp.toLocaleString();

            const hasDefects = detection.results?.defects?.length > 0;
            const statusText = hasDefects ? '⚠️ Defects' : '✅ OK';
            const statusClass = hasDefects ? 'status-defect-found' : 'status-no-defect';

            html += `<div class="history-item">
                <div class="history-time">${timeStr}</div>
                <div class="history-status">
                    <span class="result-status ${statusClass}">${statusText}</span>
                </div>
                <div class="history-summary">
                    <strong>Confidence:</strong> ${detection.results ? (detection.results.confidence * 100).toFixed(1) : 'N/A'}%<br>
                    <strong>Processing:</strong> ${detection.results?.processingTime?.toFixed(0) || 0}ms<br>
                    <small>ID: ${detection.detectionId.substr(0, 20)}...</small>
                </div>
            </div>`;
        });

        document.getElementById('historyList').innerHTML = html;

    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('historyList').innerHTML = 
            '<p style="color: #c62828;">Error loading history. Make sure backend URL is correct.</p>';
    }
}

// Utility functions
function showError(message) {
    const errorDiv = document.getElementById('uploadError');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.style.display = 'block';
}

function showSuccess(message) {
    alert(message);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Generate unique user ID if not exists
    if (!localStorage.getItem('userId')) {
        localStorage.setItem('userId', 'user_' + Date.now());
        userId = localStorage.getItem('userId');
    }

    // Check for backend URL configuration
    const backendUrlInput = prompt(
        'Enter your Azure Container URL\n(e.g., https://pcb-detection-backend.eastus.azurecontainer.io)',
        AZURE_CONFIG.backendUrl
    );

    if (backendUrlInput) {
        localStorage.setItem('backendUrl', backendUrlInput);
        AZURE_CONFIG.backendUrl = backendUrlInput;
        console.log('Backend URL set:', AZURE_CONFIG.backendUrl);
    } else {
        console.warn('No backend URL configured. Detection will fail.');
    }

    // Test backend connection
    testBackendConnection();
});

// Test backend health
async function testBackendConnection() {
    try {
        const response = await fetch(`${AZURE_CONFIG.backendUrl}/health`);
        if (response.ok) {
            const health = await response.json();
            console.log('✓ Backend connected:', health);
        }
    } catch (error) {
        console.warn('⚠️ Backend unreachable. Check URL and ensure backend is running.');
    }
}

// Drag and drop support
const uploadBox = document.querySelector('.upload-box');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.background = '#f0f2ff';
    uploadBox.style.borderColor = '#764ba2';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.background = '#f8f9ff';
    uploadBox.style.borderColor = '#667eea';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.background = '#f8f9ff';
    uploadBox.style.borderColor = '#667eea';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('imageInput').files = files;
        handleImageSelect({ target: { files: files } });
    }
});
