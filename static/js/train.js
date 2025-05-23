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
    let uploadedFileNames = []; // Changed to array to hold multiple file names
    let filepaths = []; // Added to store file paths
    let statusInterval = null;
    let isTraining = false;
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const files = dataFileInput.files;
        if (!files || files.length === 0) {
            alert('Please select at least one file');
            return;
        }
        
        // Show loading indicator
        fileInfo.innerHTML = `
            <div class="alert alert-info">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    Uploading ${files.length} file(s)...
                </div>
            </div>
        `;
        fileInfo.classList.remove('d-none');
        
        // Create FormData for multiple files
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }
        
        // Upload files
        fetch('/upload_data', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFileNames = data.filenames;
                filepaths = data.filepaths; // Store filepaths
                
                // Display file info
                let infoHtml = `<p><strong>Files:</strong> ${data.filenames.length} files uploaded</p>`;
                
                if (data.info) {
                    infoHtml += `
                        <p><strong>Combined Size:</strong> ${data.info.total_size_mb.toFixed(2)} MB</p>
                        <p><strong>Total Characters:</strong> ${data.info.total_characters.toLocaleString()}</p>
                        <p><strong>Total Words:</strong> ${data.info.total_words.toLocaleString()}</p>
                    `;
                    
                    // Display individual file info
                    if (data.file_details && data.file_details.length > 0) {
                        infoHtml += `<hr><h6>Individual File Details:</h6><ul>`;
                        data.file_details.forEach(file => {
                            infoHtml += `<li><strong>${file.filename}</strong>: ${file.size_mb.toFixed(2)} MB, ${file.characters.toLocaleString()} chars</li>`;
                        });
                        infoHtml += `</ul>`;
                    }
                    
                    // Display duplicate warning if any
                    if (data.info.duplicate_warning) {
                        infoHtml += `
                            <div class="alert alert-warning mt-2">
                                <i class="bi bi-exclamation-triangle me-2"></i>
                                ${data.info.duplicate_warning}
                            </div>
                        `;
                    }
                }
                
                fileInfo.innerHTML = `
                    <div class="alert alert-success">
                        <h6>Files Uploaded Successfully</h6>
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
        
        // Check if files have been uploaded - fixed this check
        if (!uploadedFileNames || uploadedFileNames.length === 0) {
            alert('Please upload at least one training data file first');
            return;
        }
        
        // Get form values
        const modelName = document.getElementById('model-name').value;
        const nLayer = parseInt(document.getElementById('n-layer').value);
        const nHead = parseInt(document.getElementById('n-head').value);
        const nEmbd = parseInt(document.getElementById('n-embd').value);
        const maxLength = parseInt(document.getElementById('max-length').value);
        const vocabSize = parseInt(document.getElementById('vocab-size').value);
        const useGpu = document.getElementById('use-gpu').checked;
        const fp16 = document.getElementById('fp16').checked;
        const mixedPrecision = document.getElementById('mixed-precision').checked;
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const learningRate = parseFloat(document.getElementById('learning-rate').value);
        const weightDecay = parseFloat(document.getElementById('weight-decay').value);
        const maxGradNorm = parseFloat(document.getElementById('max-grad-norm').value);
        const earlyStoppingPatience = parseInt(document.getElementById('early-stopping').value);
        const useLrScheduler = document.getElementById('use-lr-scheduler').checked;
        
        // Prepare data for the request
        const trainingData = {
            data_files: uploadedFileNames, // Send all filenames as an array
            model_name: modelName,
            n_layer: nLayer,
            n_head: nHead,
            n_embd: nEmbd,
            max_length: maxLength,
            vocab_size: vocabSize,
            use_gpu: useGpu,
            fp16: fp16,
            mixed_precision: mixedPrecision,
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate,
            weight_decay: weightDecay,
            max_grad_norm: maxGradNorm,
            early_stopping_patience: earlyStoppingPatience,
            use_lr_scheduler: useLrScheduler
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