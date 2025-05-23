from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import threading
import queue
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
import logging
from tqdm import tqdm
import tiktoken

from text_dataset import TextDataset, LLMTrainer, CharTokenizer

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
        
        # Custom training loop with status updates
        output_dir = os.path.join(app.config['MODELS_FOLDER'], training_params['model_name'])
        os.makedirs(output_dir, exist_ok=True)
        
        training_logger.log("Loading training data...")
        if trainer.tokenizer is None:
            trainer.tokenizer = trainer._create_char_tokenizer(data_path)
        
        dataset = TextDataset(data_path, trainer.tokenizer, trainer.config.n_ctx)
        dataloader = DataLoader(dataset, batch_size=training_params['batch_size'], shuffle=True)
        
        optimizer = optim.AdamW(
            trainer.model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        trainer.model.train()
        total_steps = len(dataloader) * training_params['epochs']
        global_step = 0
        total_loss = 0
        
        training_status['total_epochs'] = training_params['epochs']
        training_logger.log(f"Starting training for {training_params['epochs']} epochs...")
        
        for epoch in range(training_params['epochs']):
            if not training_status['is_training']:
                break
                
            training_status['current_epoch'] = epoch + 1
            training_logger.log(f"Epoch {epoch + 1}/{training_params['epochs']}")
            
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if not training_status['is_training']:
                    break
                
                input_ids = batch['input_ids'].to(trainer.device)
                labels = batch['labels'].to(trainer.device)
                
                optimizer.zero_grad()
                outputs = trainer.model(input_ids, labels=labels)
                loss = outputs['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), training_params.get('max_grad_norm', 1.0))
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                global_step += 1
                batch_count += 1
                
                # Update status
                training_status['current_loss'] = loss.item()
                training_status['avg_loss'] = total_loss / global_step
                training_status['progress'] = int((global_step / total_steps) * 100)
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    training_logger.log(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            training_logger.log(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint every epoch
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch + 1}')
            trainer.save_model(checkpoint_dir)
            training_logger.log(f"Checkpoint saved to {checkpoint_dir}")
        
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
            'total_parameters': sum(p.numel() for p in trainer.model.parameters())
        }
        
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        training_logger.log(f"Training completed! Final average loss: {training_status['avg_loss']:.4f}")
        training_logger.log(f"Model saved to {output_dir}")
        
    except Exception as e:
        training_logger.log(f"Training error: {str(e)}")
        
    finally:
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
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the file
        file_info = analyze_training_data(filepath)
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'filepath': filepath,
            'info': file_info
        })

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'})
    
    data = request.json
    
    # Validate required fields
    required_fields = ['data_file', 'model_name', 'epochs', 'batch_size', 'learning_rate']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Missing field: {field}'})
    
    # Prepare config and training parameters
    config = {
        'n_layer': data.get('n_layer', 12),
        'n_head': data.get('n_head', 12),
        'n_embd': data.get('n_embd', 768),
        'n_ctx': data.get('max_length', 1024),
        'vocab_size': data.get('vocab_size', 50257)
    }
    
    training_params = {
        'model_name': data['model_name'],
        'epochs': data['epochs'],
        'batch_size': data['batch_size'],
        'learning_rate': data['learning_rate'],
        'weight_decay': data.get('weight_decay', 0.01),
        'max_grad_norm': data.get('max_grad_norm', 1.0)
    }
    
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], data['data_file'])
    
    if not os.path.exists(data_path):
        return jsonify({'success': False, 'error': 'Training data file not found'})
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=train_model_async,
        args=(data_path, config, training_params)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)