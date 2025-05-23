// Training page JavaScript for LLM Training Platform

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const uploadForm = document.getElementById('upload-form');
    const dataFileInput = document.getElementById('data-file');
    const fileInfo = document.getElementById('file-info');
    const fileDetails = document.getElementById('file-details');
    const trainingForm = document.getElementById('training-form');
    const startTrainingBtn = document.getElementById('start-training-btn');
    const trainingStatus = document.getElementById('training-status');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    const viewModelBtn = document.getElementById('view-model-btn');
    const currentEpochEl = document.getElementById('current-epoch');
    const totalEpochsEl = document.getElementById('total-epochs');
    const currentLossEl = document.getElementById('current-loss');
    const avgLossEl = document.getElementById('avg-loss');
    const progressBar = document.getElementById('progress-bar');
    const trainingLogEl = document.getElementById('training-log');
    
    // Global variables
    let uploadedFileName = null;
    let statusInterval = null;
    let isTraining = false;
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = dataFileInput.files[0];
        if (!file) {
            alert('Please select a file first');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading indicator
        fileInfo.innerHTML = `
            <div class="alert alert-info">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    Uploading file...
                </div>
            </div>
        `;
        fileInfo.classList.remove('d-none');
        
        // Upload file
        fetch('/upload_data', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFileName = data.filename;
                
                // Display file info
                let infoHtml = `<p><strong>File:</strong> ${data.filename}</p>`;
                
                if (data.info) {
                    if (data.info.type === 'text') {
                        infoHtml += `
                            <p><strong>Size:</strong> ${data.info.size_mb} MB</p>
                            <p><strong>Characters:</strong> ${data.info.characters.toLocaleString()}</p>
                            <p><strong>Words:</strong> ${data.info.words.toLocaleString()}</p>
                            <p><strong>Lines:</strong> ${data.info.lines.toLocaleString()}</p>
                        `;
                    } else if (data.info.type === 'json') {
                        infoHtml += `
                            <p><strong>Size:</strong> ${data.info.size_mb} MB</p>
                            <p><strong>Records:</strong> ${data.info.records.toLocaleString()}</p>
                            <p><strong>Total Characters:</strong> ${data.info.total_characters.toLocaleString()}</p>
                            <p><strong>Total Words:</strong> ${data.info.total_words.toLocaleString()}</p>
                        `;
                    } else {
                        infoHtml += `<p><strong>Size:</strong> ${data.info.size_mb} MB</p>`;
                    }
                }
                
                fileInfo.innerHTML = `
                    <div class="alert alert-success">
                        <h6>File Uploaded Successfully</h6>
                        ${infoHtml}
                    </div>
                `;
                
                // Enable training button
                startTrainingBtn.disabled = false;
            } else {
                fileInfo.innerHTML = `
                    <div class="alert alert-danger">
                        <h6>Upload Failed</h6>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            fileInfo.innerHTML = `
                <div class="alert alert-danger">
                    <h6>Upload Failed</h6>
                    <p>An error occurred during upload</p>
                </div>
            `;
        });
    });
    
    // Handle training form submission
    trainingForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!uploadedFileName) {
            alert('Please upload a training data file first');
            return;
        }
        
        // Get form values
        const modelName = document.getElementById('model-name').value;
        const nLayer = parseInt(document.getElementById('n-layer').value);
        const nHead = parseInt(document.getElementById('n-head').value);
        const nEmbd = parseInt(document.getElementById('n-embd').value);
        const maxLength = parseInt(document.getElementById('max-length').value);
        const vocabSize = parseInt(document.getElementById('vocab-size').value);
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const learningRate = parseFloat(document.getElementById('learning-rate').value);
        const weightDecay = parseFloat(document.getElementById('weight-decay').value);
        const maxGradNorm = parseFloat(document.getElementById('max-grad-norm').value);
        
        // Prepare data for the request
        const trainingData = {
            data_file: uploadedFileName,
            model_name: modelName,
            n_layer: nLayer,
            n_head: nHead,
            n_embd: nEmbd,
            max_length: maxLength,
            vocab_size: vocabSize,
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate,
            weight_decay: weightDecay,
            max_grad_norm: maxGradNorm
        };
        
        // Start training
        fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(trainingData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show training status section
                trainingStatus.classList.remove('d-none');
                
                // Start polling for status updates
                isTraining = true;
                startStatusPolling();
                
                // Disable training form
                startTrainingBtn.disabled = true;
                Array.from(trainingForm.elements).forEach(el => {
                    if (el !== startTrainingBtn) {
                        el.disabled = true;
                    }
                });
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while starting training');
        });
    });
    
    // Handle stop training button
    stopTrainingBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to stop training? Progress will be saved, but training cannot be resumed.')) {
            fetch('/stop_training', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addLogMessage('Training stopped by user');
                    
                    // Show view model button
                    viewModelBtn.classList.remove('d-none');
                    
                    // Enable form fields
                    Array.from(trainingForm.elements).forEach(el => {
                        el.disabled = false;
                    });
                    
                    isTraining = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    });
    
    // Handle view model button
    viewModelBtn.addEventListener('click', function() {
        const modelName = document.getElementById('model-name').value;
        window.location.href = `/models?highlight=${modelName}`;
    });
    
    // Function to start polling for status updates
    function startStatusPolling() {
        // Clear any existing interval
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Initial status update
        updateTrainingStatus();
        
        // Set interval for updates
        statusInterval = setInterval(function() {
            updateTrainingStatus();
            
            // If training completes, stop polling
            if (!isTraining) {
                clearInterval(statusInterval);
            }
        }, 1000);
    }
    
    // Function to update training status
    function updateTrainingStatus() {
        fetch('/training_status')
        .then(response => response.json())
        .then(status => {
            // Update UI with status
            currentEpochEl.textContent = status.current_epoch;
            totalEpochsEl.textContent = status.total_epochs;
            currentLossEl.textContent = status.current_loss.toFixed(4);
            avgLossEl.textContent = status.avg_loss.toFixed(4);
            progressBar.style.width = `${status.progress}%`;
            
            // Update log
            updateTrainingLog(status.log);
            
            // Check if training is still active
            isTraining = status.is_training;
            
            if (!isTraining && status.current_epoch > 0) {
                // Training completed or was stopped
                viewModelBtn.classList.remove('d-none');
                
                // Enable form fields
                Array.from(trainingForm.elements).forEach(el => {
                    el.disabled = false;
                });
                
                // Clear interval
                if (statusInterval) {
                    clearInterval(statusInterval);
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Function to update training log
    function updateTrainingLog(logs) {
        if (!logs || !Array.isArray(logs)) return;
        
        trainingLogEl.innerHTML = '';
        
        logs.forEach(log => {
            addLogMessage(log);
        });
        
        // Scroll to bottom
        trainingLogEl.scrollTop = trainingLogEl.scrollHeight;
    }
    
    // Function to add a message to the log
    function addLogMessage(message) {
        const p = document.createElement('p');
        p.textContent = message;
        trainingLogEl.appendChild(p);
    }
    
    // Check if training is already in progress when page loads
    updateTrainingStatus();
});