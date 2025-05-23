// Test page JavaScript for LLM Training Platform

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const generateForm = document.getElementById('generate-form');
    const modelSelect = document.getElementById('model-select');
    const promptInput = document.getElementById('prompt');
    const maxLengthInput = document.getElementById('max-length');
    const temperatureInput = document.getElementById('temperature');
    const tempValueEl = document.getElementById('temp-value');
    const generateBtn = document.getElementById('generate-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const outputDisplay = document.getElementById('output-display');
    const copyOutputBtn = document.getElementById('copy-output');
    const historyList = document.getElementById('history-list');
    const modelInfo = document.getElementById('model-info');
    const modelDetails = document.getElementById('model-details');
    
    // Global variables
    let generationHistory = [];
    
    // Load generation history from localStorage if available
    try {
        const savedHistory = localStorage.getItem('generationHistory');
        if (savedHistory) {
            generationHistory = JSON.parse(savedHistory);
            updateHistoryList();
        }
    } catch (e) {
        console.error('Error loading generation history:', e);
    }
    
    // Check URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const modelParam = urlParams.get('model');
    
    if (modelParam) {
        // Select the model from URL parameter if it exists in the select options
        for (let i = 0; i < modelSelect.options.length; i++) {
            if (modelSelect.options[i].value === modelParam) {
                modelSelect.selectedIndex = i;
                modelSelect.dispatchEvent(new Event('change'));
                break;
            }
        }
    }
    
    // Update temperature display
    temperatureInput.addEventListener('input', function() {
        tempValueEl.textContent = this.value;
    });
    
    // Handle model selection change
    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        
        if (selectedModel) {
            // Find model details in models list
            const models = Array.from(this.options)
                .filter(opt => opt.value === selectedModel)
                .map(opt => opt.textContent);
            
            if (models.length > 0) {
                modelInfo.classList.remove('d-none');
                modelDetails.innerHTML = `<p><strong>Selected Model:</strong> ${models[0]}</p>`;
            }
        } else {
            modelInfo.classList.add('d-none');
        }
    });
    
    // Handle generate form submission
    generateForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedModel = modelSelect.value;
        const prompt = promptInput.value.trim();
        
        if (!selectedModel) {
            alert('Please select a model');
            return;
        }
        
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('d-none');
        outputDisplay.innerHTML = '';
        generateBtn.disabled = true;
        copyOutputBtn.disabled = true;
        
        // Prepare data for the request
        const generationData = {
            model_name: selectedModel,
            prompt: prompt,
            max_length: parseInt(maxLengthInput.value),
            temperature: parseFloat(temperatureInput.value)
        };
        
        // Generate text
        fetch('/generate_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(generationData)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            generateBtn.disabled = false;
            
            if (data.success) {
                // Display generated text
                outputDisplay.textContent = data.generated_text;
                copyOutputBtn.disabled = false;
                
                // Add to history
                addToHistory(prompt, data.generated_text, selectedModel);
            } else {
                outputDisplay.innerHTML = `
                    <div class="alert alert-danger">
                        <h6>Generation Failed</h6>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.classList.add('d-none');
            generateBtn.disabled = false;
            
            outputDisplay.innerHTML = `
                <div class="alert alert-danger">
                    <h6>Generation Failed</h6>
                    <p>An error occurred during text generation</p>
                </div>
            `;
        });
    });
    
    // Handle copy output button
    copyOutputBtn.addEventListener('click', function() {
        const outputText = outputDisplay.textContent;
        
        if (outputText) {
            navigator.clipboard.writeText(outputText)
                .then(() => {
                    // Show copied indicator
                    const originalText = this.textContent;
                    this.textContent = 'Copied!';
                    
                    setTimeout(() => {
                        this.textContent = originalText;
                    }, 2000);
                })
                .catch(err => {
                    console.error('Error copying text:', err);
                    alert('Failed to copy text to clipboard');
                });
        }
    });
    
    // Function to add to history
    function addToHistory(prompt, output, model) {
        const timestamp = new Date().toISOString();
        
        // Add to beginning of array
        generationHistory.unshift({
            prompt,
            output,
            model,
            timestamp
        });
        
        // Limit history to 10 items
        if (generationHistory.length > 10) {
            generationHistory = generationHistory.slice(0, 10);
        }
        
        // Save to localStorage
        try {
            localStorage.setItem('generationHistory', JSON.stringify(generationHistory));
        } catch (e) {
            console.error('Error saving generation history:', e);
        }
        
        // Update history list
        updateHistoryList();
    }
    
    // Function to update history list
    function updateHistoryList() {
        if (generationHistory.length === 0) {
            historyList.innerHTML = '<li class="list-group-item text-center text-muted">No generation history yet</li>';
            return;
        }
        
        historyList.innerHTML = '';
        
        generationHistory.forEach((item, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item history-item';
            
            // Format timestamp
            let timeString = '';
            try {
                const date = new Date(item.timestamp);
                timeString = date.toLocaleString();
            } catch (e) {
                timeString = 'Unknown time';
            }
            
            li.innerHTML = `
                <div class="d-flex justify-content-between">
                    <div class="history-prompt text-truncate">${item.prompt}</div>
                    <small class="text-muted">${timeString}</small>
                </div>
                <div class="history-output">${item.output}</div>
            `;
            
            // Add click handler to restore history item
            li.addEventListener('click', function() {
                restoreHistoryItem(index);
            });
            
            historyList.appendChild(li);
        });
    }
    
    // Function to restore history item
    function restoreHistoryItem(index) {
        const item = generationHistory[index];
        
        if (!item) return;
        
        // Restore values to form
        promptInput.value = item.prompt;
        
        // Select model if it exists
        for (let i = 0; i < modelSelect.options.length; i++) {
            if (modelSelect.options[i].value === item.model) {
                modelSelect.selectedIndex = i;
                modelSelect.dispatchEvent(new Event('change'));
                break;
            }
        }
        
        // Display the output
        outputDisplay.textContent = item.output;
        copyOutputBtn.disabled = false;
    }
});