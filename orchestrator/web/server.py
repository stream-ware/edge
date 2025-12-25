"""
Web Server for Voice Testing Interface.

Provides:
- REST API for commands
- WebSocket for real-time audio streaming
- Static HTML interface
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None

import numpy as np

logger = logging.getLogger("web")


def create_app(orchestrator=None) -> "FastAPI":
    """Create FastAPI application."""
    if FastAPI is None:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Streamware Voice Interface",
        description="Web interface for testing STT/TTS",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store orchestrator reference
    app.state.orchestrator = orchestrator
    app.state.active_connections = []
    
    # =========================================
    # REST Endpoints
    # =========================================
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve main HTML page."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse(content=get_embedded_html())
    
    @app.get("/api/status")
    async def status():
        """Get system status."""
        orch = app.state.orchestrator
        
        return {
            "status": "ok",
            "stt_available": orch.stt is not None if orch else False,
            "tts_available": orch.tts is not None if orch else False,
            "llm_available": orch.llm is not None if orch else False,
            "mqtt_connected": orch.mqtt._connected if orch and orch.mqtt else False,
            "active_ws": len(app.state.active_connections)
        }
    
    @app.post("/api/command")
    async def command(payload: dict):
        """Process text command."""
        orch = app.state.orchestrator
        if not orch:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        text = payload.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")
        
        response = await orch.process_command(text, source="web")
        return {"response": response}
    
    @app.post("/api/tts")
    async def tts(payload: dict):
        """Synthesize text to speech."""
        orch = app.state.orchestrator
        if not orch or not orch.tts:
            raise HTTPException(status_code=503, detail="TTS not available")
        
        text = payload.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")
        
        await orch.tts.speak(text)
        return {"status": "ok"}
    
    @app.get("/api/settings")
    async def get_settings():
        """Get current settings."""
        from ..settings import settings
        return settings.to_dict()
    
    # =========================================
    # WebSocket Endpoints
    # =========================================
    
    @app.websocket("/ws/audio")
    async def websocket_audio(websocket: WebSocket):
        """WebSocket for audio streaming."""
        await websocket.accept()
        app.state.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active: {len(app.state.active_connections)}")
        
        orch = app.state.orchestrator
        
        try:
            while True:
                data = await websocket.receive()
                
                if "text" in data:
                    # Text command
                    msg = json.loads(data["text"])
                    await handle_ws_message(websocket, msg, orch)
                    
                elif "bytes" in data:
                    # Audio data
                    await handle_ws_audio(websocket, data["bytes"], orch)
                    
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in app.state.active_connections:
                app.state.active_connections.remove(websocket)
    
    async def handle_ws_message(ws: WebSocket, msg: dict, orch):
        """Handle WebSocket text message."""
        msg_type = msg.get("type", "")
        
        if msg_type == "command":
            text = msg.get("text", "")
            if text and orch:
                response = await orch.process_command(text, source="websocket")
                await ws.send_json({"type": "response", "text": response})
        
        elif msg_type == "tts":
            text = msg.get("text", "")
            if text and orch and orch.tts:
                await orch.tts.speak(text)
                await ws.send_json({"type": "tts_done"})
        
        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})
    
    async def handle_ws_audio(ws: WebSocket, audio_bytes: bytes, orch):
        """Handle WebSocket audio data."""
        if not orch or not orch.stt:
            await ws.send_json({"type": "error", "message": "STT not available"})
            return
        
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Put audio in STT queue for processing
        if hasattr(orch.stt, '_audio_queue'):
            orch.stt._audio_queue.put(audio)
    
    return app


def get_embedded_html() -> str:
    """Return embedded HTML for voice testing."""
    return '''<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streamware Voice Test</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
        }
        .status-bar {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .status-item {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            background: rgba(255,255,255,0.1);
        }
        .status-item.ok { background: rgba(0, 200, 100, 0.3); }
        .status-item.error { background: rgba(200, 50, 50, 0.3); }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            margin-bottom: 16px;
            font-size: 18px;
            color: #00d4ff;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
            margin: 5px;
        }
        .btn-primary {
            background: #00d4ff;
            color: #1a1a2e;
        }
        .btn-primary:hover { background: #00b8e6; }
        .btn-danger {
            background: #ff4757;
            color: white;
        }
        .btn-danger:hover { background: #ff3344; }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .recording {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4); }
            50% { box-shadow: 0 0 0 15px rgba(255, 71, 87, 0); }
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #eee;
            font-size: 16px;
            margin-bottom: 12px;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .log {
            background: rgba(0,0,0,0.4);
            border-radius: 8px;
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 13px;
            line-height: 1.6;
        }
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .log-entry.stt { color: #00d4ff; }
        .log-entry.tts { color: #ffd93d; }
        .log-entry.cmd { color: #6bcb77; }
        .log-entry.error { color: #ff6b6b; }
        .visualizer {
            height: 60px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            margin-top: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
        }
        .bar {
            width: 4px;
            background: #00d4ff;
            border-radius: 2px;
            transition: height 0.1s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Streamware Voice Test</h1>
        
        <div class="status-bar" id="statusBar">
            <span class="status-item" id="statusWs">WebSocket: ...</span>
            <span class="status-item" id="statusStt">STT: ...</span>
            <span class="status-item" id="statusTts">TTS: ...</span>
        </div>
        
        <div class="card">
            <h2>🎙️ Voice Input</h2>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button class="btn btn-primary" id="btnRecord" onclick="toggleRecording()">
                    🎤 Start Recording
                </button>
                <button class="btn btn-danger" id="btnStop" onclick="stopRecording()" disabled>
                    ⏹️ Stop
                </button>
            </div>
            <div class="visualizer" id="visualizer">
                <!-- Audio bars will be inserted here -->
            </div>
        </div>
        
        <div class="card">
            <h2>⌨️ Text Command</h2>
            <input type="text" id="cmdInput" placeholder="Type command (e.g., 'pomoc', 'docker ps')" 
                   onkeypress="if(event.key==='Enter')sendCommand()">
            <button class="btn btn-primary" onclick="sendCommand()">Send</button>
        </div>
        
        <div class="card">
            <h2>🔊 Text-to-Speech Test</h2>
            <input type="text" id="ttsInput" placeholder="Text to speak..." value="Cześć, jestem asystentem głosowym.">
            <button class="btn btn-primary" onclick="testTTS()">Speak</button>
        </div>
        
        <div class="card">
            <h2>📜 Log</h2>
            <div class="log" id="log"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioContext = null;
        let analyser = null;
        let isRecording = false;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initVisualizer();
            connectWebSocket();
            checkStatus();
            setInterval(checkStatus, 5000);
        });
        
        function initVisualizer() {
            const viz = document.getElementById('visualizer');
            for (let i = 0; i < 32; i++) {
                const bar = document.createElement('div');
                bar.className = 'bar';
                bar.style.height = '4px';
                viz.appendChild(bar);
            }
        }
        
        function connectWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/audio`);
            
            ws.onopen = () => {
                log('WebSocket connected', 'cmd');
                updateStatus('statusWs', 'WebSocket: Connected', true);
            };
            
            ws.onclose = () => {
                log('WebSocket disconnected', 'error');
                updateStatus('statusWs', 'WebSocket: Disconnected', false);
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
            
            ws.onerror = (err) => {
                log('WebSocket error: ' + err, 'error');
            };
        }
        
        function handleMessage(msg) {
            if (msg.type === 'response') {
                log('Response: ' + msg.text, 'cmd');
            } else if (msg.type === 'transcript') {
                log('STT: ' + msg.text, 'stt');
            } else if (msg.type === 'tts_done') {
                log('TTS completed', 'tts');
            } else if (msg.type === 'error') {
                log('Error: ' + msg.message, 'error');
            }
        }
        
        async function checkStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                updateStatus('statusStt', 'STT: ' + (data.stt_available ? 'Ready' : 'N/A'), data.stt_available);
                updateStatus('statusTts', 'TTS: ' + (data.tts_available ? 'Ready' : 'N/A'), data.tts_available);
            } catch (e) {
                console.error('Status check failed:', e);
            }
        }
        
        function updateStatus(id, text, ok) {
            const el = document.getElementById(id);
            el.textContent = text;
            el.className = 'status-item ' + (ok ? 'ok' : 'error');
        }
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
                return;
            }
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 64;
                source.connect(analyser);
                
                // Start visualizer
                updateVisualizer();
                
                // Create recorder
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = async (e) => {
                    if (e.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
                        const buffer = await e.data.arrayBuffer();
                        ws.send(buffer);
                    }
                };
                
                mediaRecorder.start(100); // Send chunks every 100ms
                isRecording = true;
                
                document.getElementById('btnRecord').classList.add('recording');
                document.getElementById('btnRecord').textContent = '🔴 Recording...';
                document.getElementById('btnStop').disabled = false;
                
                log('Recording started', 'stt');
                
            } catch (err) {
                log('Microphone error: ' + err.message, 'error');
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
            }
            
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            isRecording = false;
            document.getElementById('btnRecord').classList.remove('recording');
            document.getElementById('btnRecord').textContent = '🎤 Start Recording';
            document.getElementById('btnStop').disabled = true;
            
            // Reset visualizer
            document.querySelectorAll('.bar').forEach(bar => bar.style.height = '4px');
            
            log('Recording stopped', 'stt');
        }
        
        function updateVisualizer() {
            if (!analyser || !isRecording) return;
            
            const data = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(data);
            
            const bars = document.querySelectorAll('.bar');
            bars.forEach((bar, i) => {
                const value = data[i] || 0;
                bar.style.height = Math.max(4, value / 4) + 'px';
            });
            
            requestAnimationFrame(updateVisualizer);
        }
        
        async function sendCommand() {
            const input = document.getElementById('cmdInput');
            const text = input.value.trim();
            if (!text) return;
            
            log('Command: ' + text, 'cmd');
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'command', text: text }));
            } else {
                try {
                    const resp = await fetch('/api/command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    const data = await resp.json();
                    log('Response: ' + data.response, 'cmd');
                } catch (e) {
                    log('Error: ' + e.message, 'error');
                }
            }
            
            input.value = '';
        }
        
        async function testTTS() {
            const text = document.getElementById('ttsInput').value.trim();
            if (!text) return;
            
            log('TTS: ' + text, 'tts');
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'tts', text: text }));
            } else {
                try {
                    await fetch('/api/tts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                } catch (e) {
                    log('Error: ' + e.message, 'error');
                }
            }
        }
        
        function log(message, type = '') {
            const logEl = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + type;
            entry.textContent = new Date().toLocaleTimeString() + ' | ' + message;
            logEl.appendChild(entry);
            logEl.scrollTop = logEl.scrollHeight;
        }
    </script>
</body>
</html>'''


async def run_server(orchestrator=None, host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    if uvicorn is None:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")
    
    app = create_app(orchestrator)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())
