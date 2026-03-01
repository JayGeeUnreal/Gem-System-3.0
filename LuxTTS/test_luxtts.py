import os
import torch
import soundfile as sf
from zipvoice.luxvoice import LuxTTS

# --- Configuration ---
PROMPT_AUDIO_PATH = 'earn_lucky_pitch_minus_one_samplerate_24000_short.wav' # Make sure this file exists!
OUTPUT_FILENAME = 'output.wav'
TEXT_TO_SPEAK = "Hey, what's up? I'm feeling really great if you ask me honestly!"

# Sampling Parameters (Adjust these to tune quality)
RMS = 0.01          # Volume (0.01 recommended)
T_SHIFT = 0.9       # Higher = better quality, higher WER (Word Error Rate)
NUM_STEPS = 4       # 3-4 is best for efficiency vs quality
SPEED = 1.0         # 1.0 is normal speed
REF_DURATION = 5    # Duration of reference audio to use (in seconds)

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def main():
    # 1. Check for reference audio
    if not os.path.exists(PROMPT_AUDIO_PATH):
        print(f"Error: Could not find '{PROMPT_AUDIO_PATH}'. Please place an audio file in this directory.")
        return

    # 2. Select Device
    device = get_device()
    print(f"Loading LuxTTS on {device.upper()}...")

    # 3. Load Model
    # Note: On first run, this will download weights from HuggingFace
    try:
        lux_tts = LuxTTS('YatharthS/LuxTTS', device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded. processing audio...")

    # 4. Encode the prompt
    # Note: First time might take 10s to init librosa
    try:
        encoded_prompt = lux_tts.encode_prompt(
            PROMPT_AUDIO_PATH, 
            duration=REF_DURATION, 
            rms=RMS
        )
    except Exception as e:
        print(f"Error encoding prompt: {e}")
        return

    print(f"Generating speech for text: '{TEXT_TO_SPEAK}'")

    # 5. Generate Speech
    final_wav = lux_tts.generate_speech(
        TEXT_TO_SPEAK, 
        encoded_prompt, 
        num_steps=NUM_STEPS, 
        t_shift=T_SHIFT, 
        speed=SPEED, 
        return_smooth=False
    )

    # 6. Save Audio
    # Convert PyTorch tensor to numpy array and save
    final_wav_data = final_wav.cpu().numpy().squeeze()
    sf.write(OUTPUT_FILENAME, final_wav_data, 48000)

    print(f"Success! Saved audio to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()