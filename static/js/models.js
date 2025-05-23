// Models page JavaScript for LLM Training Platform

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const modelDetailsBtns = document.querySelectorAll('.model-details-btn');
    const deleteModelBtns = document.querySelectorAll('.delete-model-btn');
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    const modelConfigDetails = document.getElementById('model-config-details');
    const trainingParamsDetails = document.getElementById('training-params-details');
    const deleteModelName = document.getElementById('delete-model-name');
    
    // Current model to delete
    let currentModelToDelete = null;
    
    // Check URL parameters for highlighting
    const urlParams = new URLSearchParams(window.location.search);
    const highlightModel = urlParams.get('highlight');
    
    if (highlightModel) {
        // Find and highlight the model row
        const modelRows = document.querySelectorAll('tbody tr');
        
        modelRows.forEach(row => {
            const modelNameCell = row.querySelector('td:first-child');
            if (modelNameCell && modelNameCell.textContent === highlightModel) {
                row.classList.add('bg-light');
                row.classList.add('fw-bold');
                
                // Scroll to the highlighted row
                setTimeout(() => {
                    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 300);
            }
        });
    }
    
    // Handle model details button clicks
    modelDetailsBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Get model info from data attribute
            const modelInfo = JSON.parse(this.dataset.modelInfo);
            
            // Clear previous details
            modelConfigDetails.innerHTML = '';
            trainingParamsDetails.innerHTML = '';
            
            // Add model configuration details
            if (modelInfo.config) {
                for (const [key, value] of Object.entries(modelInfo.config)) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${formatKey(key)}</strong></td>
                        <td>${value}</td>
                    `;
                    modelConfigDetails.appendChild(row);
                }
            }
            
            // Add training parameters details
            if (modelInfo.training_params) {
                for (const [key, value] of Object.entries(modelInfo.training_params)) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${formatKey(key)}</strong></td>
                        <td>${value}</td>
                    `;
                    trainingParamsDetails.appendChild(row);
                }
            }
        });
    });
    
    // Handle delete model button clicks
    deleteModelBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const modelName = this.dataset.modelName;
            
            // Set current model to delete
            currentModelToDelete = modelName;
            
            // Update delete confirmation modal
            deleteModelName.textContent = modelName;
            
            // Show delete confirmation modal
            const deleteModal = new bootstrap.Modal(document.getElementById('deleteModelModal'));
            deleteModal.show();
        });
    });
    
    // Handle confirm delete button click
    confirmDeleteBtn.addEventListener('click', function() {
        if (!currentModelToDelete) return;
        
        // Delete model
        fetch('/delete_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: currentModelToDelete
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Close modal
                const deleteModal = bootstrap.Modal.getInstance(document.getElementById('deleteModelModal'));
                deleteModal.hide();
                
                // Reload page to reflect changes
                window.location.reload();
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while deleting the model');
        });
    });
    
    // Helper function to format keys
    function formatKey(key) {
        return key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }
});