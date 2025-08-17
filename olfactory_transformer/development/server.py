"""Development server that works without full dependencies."""

import logging
import time
import json
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from .mock_model import MockOlfactoryTransformer, MockTokenizer


class DevelopmentHandler(BaseHTTPRequestHandler):
    """HTTP handler for development server."""
    
    def __init__(self, *args, model=None, **kwargs):
        self.model = model or MockOlfactoryTransformer()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == '/health':
            self._send_json({'status': 'healthy', 'mode': 'development'})
        elif path == '/info':
            self._send_json({
                'name': 'Olfactory Transformer Development Server',
                'version': '0.1.0-dev',
                'model': 'mock',
                'endpoints': ['/health', '/info', '/predict', '/analyze']
            })
        else:
            self._send_error(404, 'Endpoint not found')
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self._send_error(400, f'Invalid JSON: {e}')
            return
        
        if path == '/predict':
            self._handle_predict(data)
        elif path == '/analyze':
            self._handle_analyze(data)
        elif path == '/sensors':
            self._handle_sensors(data)
        else:
            self._send_error(404, 'Endpoint not found')
    
    def _handle_predict(self, data: Dict[str, Any]):
        """Handle scent prediction requests."""
        smiles = data.get('smiles')
        if not smiles:
            self._send_error(400, 'Missing SMILES string')
            return
        
        try:
            prediction = self.model.predict_scent(smiles)
            self._send_json({
                'smiles': smiles,
                'prediction': {
                    'primary_notes': prediction.primary_notes,
                    'intensity': prediction.intensity,
                    'confidence': prediction.confidence,
                    'similar_perfumes': prediction.similar_perfumes,
                    'chemical_family': prediction.chemical_family,
                    'safety_warnings': prediction.safety_warnings
                },
                'processing_time_ms': 100,  # Mock timing
                'model_version': 'mock-dev'
            })
        except Exception as e:
            self._send_error(500, f'Prediction error: {e}')
    
    def _handle_analyze(self, data: Dict[str, Any]):
        """Handle molecular analysis requests."""
        smiles = data.get('smiles')
        if not smiles:
            self._send_error(400, 'Missing SMILES string')
            return
        
        try:
            analysis = self.model.analyze_molecule(smiles)
            self._send_json(analysis)
        except Exception as e:
            self._send_error(500, f'Analysis error: {e}')
    
    def _handle_sensors(self, data: Dict[str, Any]):
        """Handle sensor-based predictions."""
        sensor_data = data.get('sensor_data')
        if not sensor_data:
            self._send_error(400, 'Missing sensor_data')
            return
        
        try:
            prediction = self.model.predict_from_sensors(sensor_data)
            self._send_json({
                'sensor_data': sensor_data,
                'prediction': {
                    'primary_notes': prediction.primary_notes,
                    'intensity': prediction.intensity,
                    'confidence': prediction.confidence,
                    'detection_method': prediction.detection_method
                },
                'processing_time_ms': 50  # Mock timing
            })
        except Exception as e:
            self._send_error(500, f'Sensor prediction error: {e}')
    
    def _send_json(self, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        error_response = json.dumps({
            'error': message,
            'code': code,
            'timestamp': time.time()
        })
        self.wfile.write(error_response.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom logging to reduce noise."""
        logging.info(f"DEV SERVER: {format % args}")


class DevelopmentServer:
    """Simple development server for testing."""
    
    def __init__(self, port=8000, host='localhost'):
        self.port = port
        self.host = host
        self.model = MockOlfactoryTransformer()
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the development server."""
        handler = lambda *args, **kwargs: DevelopmentHandler(*args, model=self.model, **kwargs)
        self.server = HTTPServer((self.host, self.port), handler)
        
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        logging.info(f"🚀 Development server started at http://{self.host}:{self.port}")
        logging.info("🔧 Available endpoints:")
        logging.info("  GET  /health - Health check")
        logging.info("  GET  /info - Server information")
        logging.info("  POST /predict - Scent prediction (requires 'smiles')")
        logging.info("  POST /analyze - Molecular analysis (requires 'smiles')")
        logging.info("  POST /sensors - Sensor prediction (requires 'sensor_data')")
    
    def stop(self):
        """Stop the development server."""
        if self.server:
            self.server.shutdown()
            logging.info("🛑 Development server stopped")
    
    def is_running(self):
        """Check if server is running."""
        return self.thread and self.thread.is_alive()


# CLI entry point for development server
def main():
    """Run development server from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Olfactory Transformer Development Server')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Start server
    server = DevelopmentServer(port=args.port, host=args.host)
    server.start()
    
    try:
        print(f"\n🧪 Olfactory Transformer Development Server")
        print(f"📍 Running at: http://{args.host}:{args.port}")
        print(f"🔧 Mode: Development (mock model)")
        print(f"⚡ Features: Basic prediction, analysis, sensor simulation")
        print(f"\n📖 Try these examples:")
        print(f"  curl http://{args.host}:{args.port}/health")
        print(f"  curl -X POST http://{args.host}:{args.port}/predict \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"smiles\": \"CCO\"}}'")
        print(f"\n🛑 Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        server.stop()


if __name__ == '__main__':
    main()