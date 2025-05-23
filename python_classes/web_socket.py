import threading
import json, datetime

# add progress bar in frontend for logging too

class WebSocketTrainingLogger():
    """Training logger that sends updates via WebSocket"""
    def __init__(self, socketio):
        self.logs = []
        self.socketio = socketio
        self.lock = threading.Lock()

    def log(self, message):
        """Log a message and emit it via WebSocket"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with self.lock:
            self.logs.append(log_entry)
            training_status['log'] = self.logs[-50:]  # Keep last 50 logs
        
        # Emit log to all connected clients
        self.socketio.emit('training_log', {'message': log_entry})
        
        # Also print to console for server-side logging
        print(log_entry)
    
    def emit_status_update(self):
        """Emit current training status via WebSocket"""
        with self.lock:
            self.socketio.emit('training_status', training_status)

