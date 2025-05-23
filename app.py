from flask_socketio import SocketIO, emit
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
from datetime import datetime
import torch
import torch.optim as optim

from python_classes.web_socket import WebSocketTrainingLogger
from python_classes.text_dataset import TextDataset, LLMTrainer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'saved_models'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global variables for training status
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'avg_loss': 0.0,
    'progress': 0,
    'log': []
}

class TrainingLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        training_status['log'] = self.logs[-50:]  # Keep last 50 logs
        print(log_entry)

training_logger = TrainingLogger()

def train_model_async(data_path, config, training_params):
    """Train model in background thread"""
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['current_epoch'] = 0
        training_status['log'] = []
        
        training_logger.log("Initializing model...")
        trainer = LLMTrainer(**config)
        
        training_logger.log(f"Model initialized with {sum(p.numel() for p in trainer.model.parameters())} parameters")
        if trainer.device.type == 'cuda':
            training_logger.log(f"Using GPU: {torch.cuda.get_device_name(trainer.device)}")
            training_logger.log(f"GPU Memory: {torch.cuda.get_device_properties(trainer.device.index).total_memory / (1024**3):.2f} GB")
        else:
            training_logger.log("Using CPU for training (enable GPU for faster training)")
        
        # Custom training loop with status updates
        output_dir = os.path.join(app.config['MODELS_FOLDER'], training_params['model_name'])
        os.makedirs(output_dir, exist_ok=True)
        
        training_logger.log("Loading training data...")
        if trainer.tokenizer is None:
            trainer.tokenizer = trainer._create_char_tokenizer(data_path)
        
        # If multiple files provided, process them all
        if isinstance(data_path, list):
            combined_dataset = None
            for path in data_path:
                training_logger.log(f"Processing file: {os.path.basename(path)}")
                if combined_dataset is None:
                    combined_dataset = TextDataset(path, trainer.tokenizer, trainer.config.n_ctx)
                else:
                    new_dataset = TextDataset(path, trainer.tokenizer, trainer.config.n_ctx)
                    # Append the new dataset (this is a simplified approach - a proper implementation would
                    # combine the datasets more efficiently and deduplicate data)
                    combined_dataset.tokens.extend(new_dataset.tokens)
                    combined_dataset.num_samples = max(1, len(combined_dataset.tokens) - combined_dataset.context_length)
            dataset = combined_dataset
            training_logger.log(f"Combined dataset created with {dataset.num_samples} samples")
        else:
            dataset = TextDataset(data_path, trainer.tokenizer, trainer.config.n_ctx)
            training_logger.log(f"Dataset created with {dataset.num_samples} samples")
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=True,
            num_workers=training_params.get('num_workers', 2),
            pin_memory=trainer.device.type == 'cuda'
        )
        
        optimizer = optim.AdamW(
            trainer.model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Add learning rate scheduler
        if training_params.get('use_lr_scheduler', False):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=len(dataloader) * training_params['epochs'],
                eta_min=training_params.get('min_lr', 1e-6)
            )
            training_logger.log("Using cosine learning rate scheduler")
        else:
            scheduler = None
        
        trainer.model.train()
        total_steps = len(dataloader) * training_params['epochs']
        global_step = 0
        total_loss = 0
        best_loss = float('inf')
        patience_counter = 0
        max_patience = training_params.get('early_stopping_patience', 0)
        
        training_status['total_epochs'] = training_params['epochs']
        training_logger.log(f"Starting training for {training_params['epochs']} epochs...")
        
        # Training metrics tracking
        training_metrics = {
            'steps': [],
            'loss': [],
            'learning_rate': [],
            'gpu_memory': [] if trainer.device.type == 'cuda' else None
        }
        
        for epoch in range(training_params['epochs']):
            if not training_status['is_training']:
                training_logger.log("Training stopped by user")
                break
                
            training_status['current_epoch'] = epoch + 1
            training_logger.log(f"Epoch {epoch + 1}/{training_params['epochs']}")
            
            epoch_loss = 0
            batch_count = 0
            
            # Track time per epoch
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                if not training_status['is_training']:
                    break
                
                # Transfer batch to device
                input_ids = batch['input_ids'].to(trainer.device)
                labels = batch['labels'].to(trainer.device)
                
                # Track GPU memory if available
                if trainer.device.type == 'cuda' and batch_idx % 10 == 0:
                    mem_allocated = torch.cuda.memory_allocated(trainer.device) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(trainer.device) / (1024**3)
                    training_metrics['gpu_memory'].append({
                        'step': global_step,
                        'allocated': mem_allocated,
                        'reserved': mem_reserved
                    })
                    training_status['gpu_memory'] = f"{mem_allocated:.2f}GB / {mem_reserved:.2f}GB"
                
                # Use mixed precision if available
                if trainer.scaler:
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = trainer.model(input_ids, labels=labels)
                        loss = outputs['loss']
                    
                    # Scale gradients and optimize
                    trainer.scaler.scale(loss).backward()
                    trainer.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), training_params.get('max_grad_norm', 1.0))
                    trainer.scaler.step(optimizer)
                    trainer.scaler.update()
                else:
                    optimizer.zero_grad()
                    outputs = trainer.model(input_ids, labels=labels)
                    loss = outputs['loss']
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), training_params.get('max_grad_norm', 1.0))
                    optimizer.step()
                
                # Update learning rate if scheduler is enabled
                if scheduler:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    training_metrics['learning_rate'].append({
                        'step': global_step,
                        'lr': current_lr
                    })
                    training_status['learning_rate'] = current_lr
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                global_step += 1
                batch_count += 1
                
                # Update training metrics
                training_metrics['steps'].append(global_step)
                training_metrics['loss'].append(loss.item())
                
                # Update status
                training_status['current_loss'] = loss.item()
                training_status['avg_loss'] = total_loss / global_step
                training_status['progress'] = int((global_step / total_steps) * 100)
                training_status['steps_per_sec'] = batch_count / (time.time() - epoch_start_time)
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    current_time = time.time()
                    steps_per_sec = 10 / (current_time - (epoch_start_time + (batch_idx - 10) / batch_count * (current_time - epoch_start_time))) if batch_idx > 0 else 0
                    remaining_batches = len(dataloader) - batch_idx
                    est_time_remaining = remaining_batches / steps_per_sec if steps_per_sec > 0 else 0
                    
                    training_logger.log(
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Speed: {steps_per_sec:.2f} steps/sec, "
                        f"Est. time: {est_time_remaining/60:.1f} min remaining"
                    )
            
            # End of epoch processing
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_time = time.time() - epoch_start_time
            
            training_logger.log(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. "
                f"Average loss: {avg_epoch_loss:.4f}"
            )
            
            # Early stopping check
            if max_patience > 0:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                    training_logger.log(f"New best loss: {best_loss:.4f}")
                    
                    # Save best model
                    best_model_dir = os.path.join(output_dir, 'best_model')
                    trainer.save_model(best_model_dir)
                    training_logger.log(f"Best model saved to {best_model_dir}")
                else:
                    patience_counter += 1
                    training_logger.log(f"No improvement for {patience_counter}/{max_patience} epochs")
                    
                    if patience_counter >= max_patience:
                        training_logger.log("Early stopping triggered!")
                        break
            
            # Save checkpoint every epoch
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch + 1}')
            trainer.save_model(checkpoint_dir)
            training_logger.log(f"Checkpoint saved to {checkpoint_dir}")
            
            # Save training metrics
            with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
                json.dump(training_metrics, f)
        
        # Save final model
        final_model_dir = os.path.join(output_dir, 'final_model')
        trainer.save_model(final_model_dir)
        
        # Save training info
        training_info = {
            'model_name': training_params['model_name'],
            'config': config,
            'training_params': training_params,
            'final_loss': training_status['avg_loss'],
            'completion_time': datetime.now().isoformat(),
            'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'training_time': time.time() - epoch_start_time if 'epoch_start_time' in locals() else None,
            'early_stopped': patience_counter >= max_patience if max_patience > 0 else False
        }
        
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        training_logger.log(f"Training completed! Final average loss: {training_status['avg_loss']:.4f}")
        training_logger.log(f"Model saved to {output_dir}")
        
    except Exception as e:
        import traceback
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        training_logger.log(error_msg)
        
    finally:
        # Clean up and release GPU memory
        if 'trainer' in locals() and hasattr(trainer, 'device') and trainer.device.type == 'cuda':
            try:
                trainer.model.to('cpu')
                torch.cuda.empty_cache()
                training_logger.log("GPU memory cleared")
            except:
                pass
        
        training_status['is_training'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/test')
def test_page():
    # Get available models
    models = []
    if os.path.exists(app.config['MODELS_FOLDER']):
        for model_dir in os.listdir(app.config['MODELS_FOLDER']):
            model_path = os.path.join(app.config['MODELS_FOLDER'], model_dir)
            if os.path.isdir(model_path):
                info_file = os.path.join(model_path, 'training_info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    models.append({
                        'name': model_dir,
                        'info': info
                    })
    
    return render_template('test.html', models=models)

@app.route('/models')
def models_page():
    # Get all saved models with their info
    models = []
    if os.path.exists(app.config['MODELS_FOLDER']):
        for model_dir in os.listdir(app.config['MODELS_FOLDER']):
            model_path = os.path.join(app.config['MODELS_FOLDER'], model_dir)
            if os.path.isdir(model_path):
                info_file = os.path.join(model_path, 'training_info.json')
                model_info = {
                    'name': model_dir,
                    'path': model_path,
                    'size': get_folder_size(model_path),
                    'created': datetime.fromtimestamp(os.path.getctime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        training_info = json.load(f)
                    model_info.update(training_info)
                
                models.append(model_info)
    
    return render_template('models.html', models=models)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected'})
    
    # Store file paths and info
    file_paths = []
    file_details = []
    total_size = 0
    total_characters = 0
    total_words = 0
    content_hashes = set()  # For deduplication check
    duplicate_content = False
    
    try:
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_paths.append(filepath)
            
            # Analyze file and calculate hash for deduplication
            file_info = analyze_training_data(filepath)
            file_size = os.path.getsize(filepath)
            total_size += file_size
            
            # Basic deduplication check based on file content
            with open(filepath, 'rb') as f:
                content = f.read()
                content_hash = hash(content)
                if content_hash in content_hashes:
                    duplicate_content = True
                content_hashes.add(content_hash)
            
            # Calculate text statistics
            if filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chars = len(content)
                    words = len(content.split())
                    total_characters += chars
                    total_words += words
                    
                    file_details.append({
                        'filename': filename,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'characters': chars,
                        'words': words
                    })
            
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    total_text = ''
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            total_text += item['text'] + ' '
                        elif isinstance(item, str):
                            total_text += item + ' '
                    
                    chars = len(total_text)
                    words = len(total_text.split())
                    total_characters += chars
                    total_words += words
                    
                    file_details.append({
                        'filename': filename,
                        'size_mb': round(file_size / 1024 / 1024, 2),
                        'characters': chars,
                        'words': words,
                        'records': len(data)
                    })
        
        # Create combined info object
        combined_info = {
            'total_size_mb': total_size / 1024 / 1024,
            'total_characters': total_characters,
            'total_words': total_words,
            'file_count': len(file_paths),
            'duplicate_warning': "Some files appear to have duplicate content. The model will skip redundant training data." if duplicate_content else None
        }
        
        return jsonify({
            'success': True, 
            'filenames': [os.path.basename(path) for path in file_paths],
            'filepaths': file_paths,
            'info': combined_info,
            'file_details': file_details
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Update start_training route to handle multiple files
@app.route('/start_training', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'})
    
    data = request.json
    
    # Validate required fields
    required_fields = ['data_files', 'model_name', 'epochs', 'batch_size', 'learning_rate']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Missing field: {field}'})
    
    # Prepare config and training parameters
    config = {
        'n_layer': int(data.get('n_layer', 12)),
        'n_head': int(data.get('n_head', 12)),
        'n_embd': int(data.get('n_embd', 768)),
        'n_ctx': int(data.get('max_length', 1024)),
        'vocab_size': int(data.get('vocab_size', 50257)),
        'use_gpu': data.get('use_gpu', True),
        'fp16': data.get('fp16', False),
        'device_id': int(data.get('device_id', 0)),
        'mixed_precision': data.get('mixed_precision', True)
    }
    
    training_params = {
        'model_name': data['model_name'],
        'epochs': int(data['epochs']),
        'batch_size': int(data['batch_size']),
        'learning_rate': float(data['learning_rate']),
        'weight_decay': float(data.get('weight_decay', 0.01)),
        'max_grad_norm': float(data.get('max_grad_norm', 1.0)),
        'use_lr_scheduler': data.get('use_lr_scheduler', False),
        'min_lr': float(data.get('min_lr', 1e-6)),
        'early_stopping_patience': int(data.get('early_stopping_patience', 0)),
        'num_workers': int(data.get('num_workers', 2))
    }
    
    # Process single file or multiple files
    if isinstance(data['data_files'], list):
        data_paths = [os.path.join(app.config['UPLOAD_FOLDER'], file) for file in data['data_files']]
        # Verify all files exist
        for path in data_paths:
            if not os.path.exists(path):
                return jsonify({'success': False, 'error': f'Training data file not found: {os.path.basename(path)}'})
    else:
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], data['data_files'])
        if not os.path.exists(data_path):
            return jsonify({'success': False, 'error': 'Training data file not found'})
        data_paths = data_path
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=train_model_async,
        args=(data_paths, config, training_params)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_status
    training_status['is_training'] = False
    training_logger.log("Training stopped by user")
    return jsonify({'success': True, 'message': 'Training stopped'})

@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    model_name = data.get('model_name')
    prompt = data.get('prompt', '') 
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.8)
    
    try:
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_name, 'final_model')
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'Model not found'})
        
        # Load and use model
        trainer = LLMTrainer()
        trainer.load_model(model_path)
        
        if hasattr(trainer.tokenizer, 'encode'):
            input_ids = torch.tensor(trainer.tokenizer.encode(prompt), device=trainer.device)
        else:
            return jsonify({'success': False, 'error': 'Tokenizer not available'})
        
        generated = trainer.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature
        )
        
        if hasattr(trainer.tokenizer, 'decode'):
            generated_text = trainer.tokenizer.decode(generated.tolist())
            return jsonify({
                'success': True, 
                'generated_text': generated_text,
                'prompt': prompt
            })
        else:
            return jsonify({'success': False, 'error': 'Cannot decode generated tokens'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_model', methods=['POST'])
def delete_model():
    model_name = request.json.get('model_name')
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name required'})
    
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
    
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
        return jsonify({'success': True, 'message': f'Model {model_name} deleted'})
    else:
        return jsonify({'success': False, 'error': 'Model not found'})

@app.route('/download_model/<model_name>')
def download_model(model_name):
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
    
    if not os.path.exists(model_path):
        flash('Model not found', 'error')
        return redirect(url_for('models_page'))
    
    # Create a zip file of the model
    import zipfile
    import tempfile
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_path)
                zipf.write(file_path, arcname)
    
    return send_file(temp_zip.name, as_attachment=True, download_name=f'{model_name}.zip')

def analyze_training_data(filepath):
    """Analyze training data file and return information"""
    try:
        file_size = os.path.getsize(filepath)
        
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'type': 'text',
                'size_mb': round(file_size / 1024 / 1024, 2),
                'characters': len(content),
                'words': len(content.split()),
                'lines': len(content.split('\n'))
            }
        
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                total_text = ''
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        total_text += item['text'] + ' '
                    elif isinstance(item, str):
                        total_text += item + ' '
                
                return {
                    'type': 'json',
                    'size_mb': round(file_size / 1024 / 1024, 2),
                    'records': len(data),
                    'total_characters': len(total_text),
                    'total_words': len(total_text.split())
                }
        
        return {
            'type': 'unknown',
            'size_mb': round(file_size / 1024 / 1024, 2)
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'error': str(e)
        }

def get_folder_size(folder_path):
    """Get the size of a folder in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return round(total_size / 1024 / 1024, 2)

socketio = SocketIO(app, cors_allowed_origins="*")

# Replace the old logger with the WebSocket-enabled one
training_logger = WebSocketTrainingLogger(socketio, training_status)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('training_status', training_status)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('get_training_status')
def handle_get_status():
    """Send current training status to the client"""
    emit('training_status', training_status)

@socketio.on('get_full_log')
def handle_get_full_log():
    """Send full training log to the client"""
    emit('full_log', {'log': training_logger.logs})

# Update the app.run to use socketio
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)