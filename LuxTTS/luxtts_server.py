import os
import sys
import io
import random
import configparser
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import soundfile as sf
import numpy as np
import torch

# --- 1. Dynamic Path Resolution ---
# This script is in: .../Gem-System/LuxTTS
SCRIPT_DIR = Path(__file__).resolve().parent
# Parent is: .../Gem-System
PARENT_DIR = SCRIPT_DIR.parent

# Path to your settings file
CONFIG_PATH = PARENT_DIR / "mcp_settings.ini"

# Path to the voices folder in the other model's directory
VOICES_DIR = PARENT_DIR / "StyleTTS2" / "voices"

# Ensure Python can see the 'zipvoice' folder inside the LuxTTS directory
sys.path.append(str(SCRIPT_DIR))

# --- 2. LuxTTS Imports ---
try:
    from zipvoice.luxvoice import LuxTTS
except ImportError:
    print(f"ERROR: Could not find 'zipvoice' folder in {SCRIPT_DIR}")
    sys.exit(1)

# --- 3. Load Configuration ---
config = configparser.ConfigParser()
if not CONFIG_PATH.exists():
    sys.exit(f"FATAL ERROR: Configuration file not found at '{CONFIG_PATH}'")

config.read(str(CONFIG_PATH))

try:
    # LuxTTS Sampling Parameters
    NUM_STEPS = config.getint('TTS', 'num_steps', fallback=4)
    T_SHIFT = config.getfloat('TTS', 't_shift', fallback=0.9)
    SPEED = config.getfloat('TTS', 'speed', fallback=1.0)
    RMS = config.getfloat('TTS', 'rms', fallback=0.01)
    REF_DURATION = config.getint('TTS', 'ref_duration', fallback=5)
    
    # Audio Setup
    SAMPLE_RATE = 48000
    SEED = config.getint('TTS', 'seed', fallback=42)
    
    # Reference Voice: Find filename in INI, locate it in StyleTTS2/voices
    voice_filename = config.get('TTS', 'reference_voice')
    REFERENCE_VOICE_PATH = str(VOICES_DIR / voice_filename)
    
    # Server Setup: Extract Host and Port from [StyleTTS] tts_url
    tts_url = config.get('StyleTTS', 'tts_url', fallback='http://127.0.0.1:13300/tts')
    parsed_url = urlparse(tts_url)
    
    SERVER_HOST = parsed_url.hostname or '127.0.0.1'
    SERVER_PORT = parsed_url.port or 13300
    SERVER_DEBUG = config.getboolean('Server', 'debug', fallback=False)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception as e:
    sys.exit(f"FATAL ERROR: Configuration error in '{CONFIG_PATH}': {e}")

# --- 4. Helper for Reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- 5. Global LuxTTS Initialization ---
lux_model = None
global_encode_dict = None

def initialize_luxtts():
    global lux_model, global_encode_dict
    print(f"--- Initializing LuxTTS on {DEVICE.upper()} ---")
    print(f"Reading Config: {CONFIG_PATH}")
    print(f"Using Voice: {REFERENCE_VOICE_PATH}")
    
    if not os.path.exists(REFERENCE_VOICE_PATH):
        print(f"ERROR: Reference voice file not found at: {REFERENCE_VOICE_PATH}")
        sys.exit(1)

    try:
        lux_model = LuxTTS('YatharthS/LuxTTS', device=DEVICE)
        
        print(f"Encoding reference voice...")
        global_encode_dict = lux_model.encode_prompt(
            REFERENCE_VOICE_PATH, 
            duration=REF_DURATION, 
            rms=RMS
        )
        print("LuxTTS initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        sys.exit(1)

# --- 6. Flask Web Server ---
app = Flask(__name__)
CORS(app)

@app.route('/tts', methods=['POST', 'PUT'])
def tts_endpoint():
    set_seed(SEED)
    print("\n--- New TTS Request Received ---")
   
    data = request.get_json()
    if not data or 'chatmessage' not in data:
        return jsonify({"error": "Missing 'chatmessage' in JSON payload"}), 400
        
    text_to_speak = data.get('chatmessage')
    print(f"Synthesizing: '{text_to_speak[:100]}...'")

    try:
        # Generate Audio
        final_wav = lux_model.generate_speech(
            text_to_speak, 
            global_encode_dict, 
            num_steps=NUM_STEPS, 
            t_shift=T_SHIFT, 
            speed=SPEED
        )
        
        # Convert torch tensor to numpy array
        audio_data = final_wav.numpy().squeeze()
        
        # Save output for watcher_to_face.py in the main Gem-System directory
        output_filepath = str(PARENT_DIR / "server_output.wav")
        sf.write(output_filepath, audio_data, SAMPLE_RATE)
        print(f"Saved audio to: {output_filepath}")

        # Stream back to requester
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return Response(buffer, mimetype='audio/wav')

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_luxtts()
    print(f"\n--- LuxTTS Server listening on http://{SERVER_HOST}:{SERVER_PORT} ---")
    print(f"Waiting for requests at {tts_url}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG)