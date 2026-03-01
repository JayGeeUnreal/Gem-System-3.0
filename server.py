import os

# --- 1. INITIAL LOGGING & WINDOWS SETUP (MUST BE AT ABSOLUTE TOP) ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONUTF8"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import sys
import httpx
import asyncio
import configparser
import threading
import re
import inspect
import json
import contextlib
import gc
import traceback
import io
import numpy as np
import soundfile as sf
import subprocess
from urllib.parse import quote
import gradio as gr
from quart import Quart, request, jsonify
from ollama import AsyncClient
from hypercorn.config import Config
from hypercorn.asyncio import serve

# --- 2. WINDOWS-NATIVE TRAINING & LORA IMPORTS ---
import torch
HAS_TRAINER = False
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig, 
        TrainingArguments, 
        Trainer, 
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from datasets import Dataset
    HAS_TRAINER = True
    print("[+] Training & LoRA libraries loaded.")
except ImportError:
    print("[!] Training libraries not found. Knowledge Bank will be limited.")

# --- 3. CONFIGURATION MANAGEMENT ---
INI_FILE = 'mcp_settings.ini'
config_parser = configparser.ConfigParser(interpolation=None)

def load_settings():
    config_parser.read(INI_FILE, encoding='utf-8')
    
    # Ensure all sections exist to prevent startup crashes
    for section in['MCP', 'SystemPrompt', 'Assistant', 'SocialStream', 'StyleTTS']:
        if section not in config_parser:
            config_parser[section] = {}

    offline_val = config_parser.get('Assistant', 'offline_mode', fallback='True')
    is_offline = (offline_val == 'True')
    
    # Apply Offline Mode to Environment
    os.environ["HF_HUB_OFFLINE"] = "1" if is_offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if is_offline else "0"
    
    print(f"\n[!] SYSTEM STATUS: Offline Mode is {'ON' if is_offline else 'OFF'}")
    
    return {
        # LLM Settings
        "llm_model": config_parser.get('MCP', 'llm_choice', fallback='gemma3:4b'),
        "host": config_parser.get('MCP', 'host', fallback='127.0.0.1'),
        "port": config_parser.getint('MCP', 'port', fallback=5000),
        "temperature": config_parser.getfloat('MCP', 'temperature', fallback=0.85),
        "top_p": config_parser.getfloat('MCP', 'top_p', fallback=0.9),
        "repeat_penalty": config_parser.getfloat('MCP', 'repeat_penalty', fallback=1.2),
        "num_predict": config_parser.getint('MCP', 'num_predict', fallback=120),
        
        # SystemPrompt 
        "prompt": config_parser.get('SystemPrompt', 'prompt'),
        
        # Assistant / Voice Logic
        "wake_words": config_parser.get('Assistant', 'wake_words', fallback='Gem, Jen, Jim, gem'),
        "voice_engine": config_parser.get('Assistant', 'voice_engine', fallback='F5-TTS'),
        "voice_ref_path": config_parser.get('Assistant', 'voice_ref_path', fallback='gem_voice_sample.wav'),
        "voice_ref_text": config_parser.get('Assistant', 'voice_ref_text', fallback=''),
        "voice_steps": config_parser.getint('Assistant', 'voice_steps', fallback=64),
        "voice_enabled": config_parser.getboolean('Assistant', 'voice_enabled', fallback=True),
        "use_knowledge": config_parser.getboolean('Assistant', 'use_knowledge', fallback=False),
        "offline_mode": is_offline,
        
        # StyleTTS Section
        "styletts_url": config_parser.get('StyleTTS', 'tts_url', fallback='http://127.0.0.1:13300/tts'),
        "styletts_boost": config_parser.getfloat('StyleTTS', 'volume_boost', fallback=1.5),
        "styletts_enabled": config_parser.getboolean('StyleTTS', 'enabled', fallback=True),
        
        # SocialStream
        "ssn_session": config_parser.get('SocialStream', 'session_id', fallback=""),
        "ssn_api": config_parser.get('SocialStream', 'api_url', fallback="https://io.socialstream.ninja"),
        "ssn_enabled": config_parser.getboolean('SocialStream', 'enabled', fallback=True),
    }

current_conf = load_settings()

# --- 4. VOICE ENGINE & VRAM UTILS ---
voice_engine = None
engine_type = "NONE"

def flush_vram():
    """Forcefully clears GPU memory to make room for Gem's brain."""
    global voice_engine, LOCAL_MODEL
    print("[*] Flushing VRAM...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return "✅ VRAM Flush complete! Na ka."

def init_voice_engine():
    global voice_engine, engine_type
    etype = current_conf.get('voice_engine', 'F5-TTS')
    
    if voice_engine is not None and engine_type == etype:
        return

    print(f"[*] Initializing {etype} Voice Engine...")
    voice_engine = None
    flush_vram()

    try:
        # Hide the "Download Vocos" spam
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            if etype in["F5-TTS", "E2-TTS"]:
                from f5_tts.api import F5TTS
                # Match newest library version keyword for model choice
                target_model = "F5TTS_v1_Base" if etype == "F5-TTS" else "E2TTS_Base"
                voice_engine = F5TTS(model=target_model) 
                
                # Apply FP16 optimization if possible
                if torch.cuda.is_available():
                    internal = getattr(voice_engine, 'model', getattr(voice_engine, 'f5tts', None))
                    if internal:
                        try: internal.to(device="cuda", dtype=torch.float16)
                        except: pass
                engine_type = etype

            elif etype == "StyleTTS2":
                engine_type = "StyleTTS2"

            elif etype == "LuxTTS":
                sys.path.append(os.path.join(os.getcwd(), "LuxTTS"))
                from zipvoice.luxvoice import LuxTTS
                voice_engine = LuxTTS('YatharthS/LuxTTS', device="cuda")
                engine_type = "LuxTTS"
        
        print(f"[+] {etype} loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load {etype}: {e}")
        engine_type = "NONE"

def clean_text_for_tts(text):
    """Remove special characters that cause robotic glitches."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('*', '').replace('_', '').replace('#', '').replace('"', '')
    return text.strip()

def speak_cloned_text(text):
    if not current_conf['voice_enabled'] or engine_type == "NONE":
        return
    
    text = clean_text_for_tts(text)
    if not text: return

    try:
        import sounddevice as sd
        
        # --- ENGINE 1: StyleTTS2 (MICROSERVICE) ---
        if engine_type == "StyleTTS2":
            payload = {"chatmessage": text}
            response = httpx.post(current_conf['styletts_url'], json=payload, timeout=30.0)
            if response.status_code == 200:
                data, fs = sf.read(io.BytesIO(response.content))
                # Apply Volume Boost from settings
                sd.play(data * current_conf['styletts_boost'], fs)
                sd.wait()
            else:
                print(f"[!] StyleTTS Microservice Error: {response.status_code}")

        # --- ENGINE 2: F5/E2-TTS (LOCAL) ---
        elif engine_type in["F5-TTS", "E2-TTS"]:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ref_path = os.path.join(base_dir, "voices", current_conf['voice_ref_path'])
            
            if not os.path.exists(ref_path):
                print(f"[!] Voice ref not found: {ref_path}")
                return

            # Smart parameter detection
            sig = inspect.signature(voice_engine.infer)
            kwargs = {
                "ref_file": ref_path, 
                "ref_text": current_conf['voice_ref_text'][:250], 
                "gen_text": text
            }
            if "n_steps" in sig.parameters: kwargs["n_steps"] = current_conf['voice_steps']
            elif "steps" in sig.parameters: kwargs["steps"] = current_conf['voice_steps']

            wav, sr, _ = voice_engine.infer(**kwargs)
            if torch.is_tensor(wav): wav = wav.cpu().numpy()
            
            # Padding to prevent cutoff
            padded = np.concatenate([wav.flatten(), np.zeros(int(sr * 0.4))])
            sd.play(padded, sr)
            sd.wait()

        # --- ENGINE 3: LuxTTS (LOCAL) ---
        elif engine_type == "LuxTTS":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ref_path = os.path.join(base_dir, "voices", current_conf['voice_ref_path'])
            prompt_speech = voice_engine.encode_prompt(ref_path)
            audio = voice_engine.generate_speech(text, prompt_speech, num_steps=current_conf['voice_steps'])
            sd.play(audio.squeeze().cpu().numpy(), 48000)
            sd.wait()

    except Exception as e:
        print(f"[!] TTS Playback Error: {e}")

def launch_styletts2():
    """Launches the external StyleTTS2 Server batch file in a new window."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        bat_path = os.path.join(base_dir, "start_scripts", "Start_StyleTTS2.bat")
        
        if not os.path.exists(bat_path):
            return f"❌ Error: Batch file not found at {bat_path}"
            
        # CREATE_NEW_CONSOLE (0x00000010) opens it in a detached CMD window
        subprocess.Popen(
            [bat_path], 
            cwd=os.path.dirname(bat_path), 
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return "✅ StyleTTS2 Server is starting in a new window..."
    except Exception as e:
        return f"❌ Failed to launch StyleTTS2: {e}"

# --- 5. KNOWLEDGE TRAINER (JSONL) ---
def teach_gem(file_path):
    if not HAS_TRAINER: return "❌ Training libraries missing."
    if not file_path: return "❌ Upload a JSONL file first."
    if current_conf['offline_mode']: return "❌ Offline Mode is ON. Turn it OFF and hit Save first."

    try:
        flush_vram()
        data =[]
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        
        dataset = Dataset.from_list(data)
        model_id = "google/gemma-2-2b-it"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        def tok_fn(x):
            texts =[f"<start_of_turn>user\n{i}<end_of_turn>\n<start_of_turn>model\n{o}<end_of_turn>" for i, o in zip(x["instruction"], x["output"])]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = dataset.map(tok_fn, batched=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

        trainer = Trainer(
            model=model, args=TrainingArguments(output_dir="./gem_knowledge_chip", per_device_train_batch_size=1, gradient_accumulation_steps=4, max_steps=60, fp16=True, report_to="none"),
            train_dataset=tokenized_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        trainer.train()
        model.save_pretrained("./gem_knowledge_chip")
        return "✅ Persona Chip Trained! Toggle 'Use Knowledge Chip' and hit Save."
    except Exception as e:
        return f"❌ Training Error: {str(e)}"

# --- 6. LLM LOGIC (LOCAL CHIP VS OLLAMA) ---
LOCAL_MODEL = None
LOCAL_TOKENIZER = None

async def get_gem_response(user_text):
    global LOCAL_MODEL, LOCAL_TOKENIZER
    chip_path = "./gem_knowledge_chip"
    
    if current_conf['use_knowledge'] and os.path.exists(chip_path):
        try:
            if LOCAL_MODEL is None:
                print("[*] Loading Local Knowledge Brain...")
                base_id = "google/gemma-2-2b-it"
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
                LOCAL_MODEL = PeftModel.from_pretrained(base, chip_path)
                LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(base_id)

            formatted = f"<start_of_turn>user\nInstruction: {current_conf['prompt']}\nUser says: {user_text}<end_of_turn>\n<start_of_turn>model\n"
            inputs = LOCAL_TOKENIZER(formatted, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = LOCAL_MODEL.generate(**inputs, max_new_tokens=current_conf['num_predict'], temperature=current_conf['temperature'], do_sample=True, pad_token_id=LOCAL_TOKENIZER.eos_token_id)
            
            # Decoding only the new part
            reply = LOCAL_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            return reply
        except Exception as e:
            print(f"[!] Local Brain Error: {e}")

    try:
        response = await ollama_client.chat(
            model=current_conf['llm_model'],
            messages=[{'role': 'system', 'content': current_conf['prompt']}, {'role': 'user', 'content': user_text}],
            options={"temperature": current_conf['temperature'], "repeat_penalty": current_conf['repeat_penalty'], "num_predict": current_conf['num_predict']}
        )
        return response['message']['content']
    except Exception as e:
        print(f"[!] Ollama Error: {e}")
        return "Brain too hot honey. na ka!"

# --- 7. SERVER (QUART) ---
app = Quart(__name__)
ollama_client = AsyncClient()

@app.route('/chat', methods=['POST'])
async def chat_webhook():
    data = await request.get_json(force=True)
    user_msg = data.get('chatmessage') or data.get('message') or data.get('text') or ""
    author = data.get('chatname') or data.get('author') or "User"
    
    wake_list = [w.strip().lower() for w in current_conf['wake_words'].split(',')]
    if not any(word in user_msg.lower() for word in wake_list):
        return jsonify({"status": "ignored"})

    print(f"[*] {author}: {user_msg}")
    reply = await get_gem_response(user_msg)
    
    if reply:
        if reply.lower().startswith("gem:"): reply = reply[4:].strip()
        print(f"[>] Gem: {reply}")
        
        # Notify Social Stream Ninja Cloud
        if current_conf['ssn_enabled']:
            url = f"{current_conf['ssn_api']}/{current_conf['ssn_session']}/sendChat/null/{quote(reply)}"
            async with httpx.AsyncClient() as client:
                try: await client.get(url, timeout=15.0)
                except: pass

        threading.Thread(target=speak_cloned_text, args=(reply,), daemon=True).start()
        return jsonify({"status": "success", "reply": reply})
    return jsonify({"status": "error"})

def run_quart():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    h_config = Config()
    h_config.bind =[f"{current_conf['host']}:{current_conf['port']}"]
    loop.run_until_complete(serve(app, h_config, shutdown_trigger=lambda: asyncio.Future()))

# --- 8. DASHBOARD (GRADIO) ---
with gr.Blocks(title="Gem Master Control", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Gem-AI-System 3.0")
    with gr.Tabs():
        with gr.Tab("🧠 AI & Persona"):
            with gr.Row():
                with gr.Column():
                    model_in = gr.Textbox(label="Ollama Model (Isnt used when Use Knowledge Chip is Enabled)", value=current_conf['llm_model'])
                    temp_in = gr.Slider(0, 2, value=current_conf['temperature'], label="Temperature")
                    penalty_in = gr.Slider(1, 2, value=current_conf['repeat_penalty'], step=0.1, label="Repeat Penalty")
                    offline_in = gr.Checkbox(label="Force Offline Mode", value=current_conf['offline_mode'])
                    flush_btn = gr.Button("🧹 Flush VRAM (Cleanup)", variant="secondary")
                prompt_in = gr.TextArea(label="System Prompt", value=current_conf['prompt'], lines=10)
        
        with gr.Tab("🎙️ Voice & StyleTTS"):
            engine_in = gr.Dropdown(["F5-TTS", "E2-TTS", "StyleTTS2", "LuxTTS"], label="Engine", value=current_conf['voice_engine'])
            
            # ---
            with gr.Row():
                stts_url_in = gr.Textbox(label="StyleTTS2 URL", value=current_conf['styletts_url'], scale=3)
                start_stts_btn = gr.Button("🚀 Start StyleTTS2 Server", scale=1)
                stts_boost_in = gr.Slider(0.5, 3.0, value=current_conf['styletts_boost'], step=0.1, label="Volume Boost", scale=2)
                
            with gr.Row():
                ref_in = gr.Textbox(label="Sample Name (Local)", value=current_conf['voice_ref_path'])
                steps_in = gr.Slider(16, 128, value=current_conf['voice_steps'], step=8, label="Quality Steps")
            ref_text_in = gr.Textbox(label="Ref Text (Exactly what is said in wav)", value=current_conf['voice_ref_text'])
            voice_on = gr.Checkbox(label="Enable Voice Output", value=current_conf['voice_enabled'])
            wake_in = gr.Textbox(label="Wake Words", value=current_conf['wake_words'])

        with gr.Tab("📚 Knowledge Bank"):
            doc_input = gr.File(label="Upload .txt (JSONL format)")
            train_btn = gr.Button("🔥 Start Training")
            knowledge_on = gr.Checkbox(label="Use Knowledge Chip (Uses gemma-2-2b-it))", value=current_conf['use_knowledge'])
            train_status = gr.Textbox(label="Status", value="Ready")

        with gr.Tab("📡 Stream Settings"):
            ssn_sid = gr.Textbox(label="Ninja Session ID", value=current_conf['ssn_session'])
            ssn_url = gr.Textbox(label="Ninja API URL", value=current_conf['ssn_api'])
            ssn_on = gr.Checkbox(label="Enable Stream Output", value=current_conf['ssn_enabled'])

    save_btn = gr.Button("💾 Save All Settings", variant="primary")
    status_msg = gr.Textbox(label="System Status", value="Ready", interactive=False)
    
    def save_logic(m, temp, pen, off, p, k_on, e, surl, sboost, r, steps, r_text, v, w, sid, api, son):
        global current_conf, LOCAL_MODEL
        config_parser['MCP']['llm_choice'] = m; config_parser['MCP']['temperature'] = str(temp)
        config_parser['MCP']['repeat_penalty'] = str(pen); config_parser['Assistant']['offline_mode'] = str(off)
        config_parser['SystemPrompt']['prompt'] = p; config_parser['Assistant']['use_knowledge'] = str(k_on)
        config_parser['Assistant']['voice_engine'] = e; config_parser['StyleTTS']['tts_url'] = surl
        config_parser['Assistant']['voice_ref_path'] = r; config_parser['Assistant']['voice_steps'] = str(int(steps))
        config_parser['Assistant']['voice_ref_text'] = r_text; config_parser['Assistant']['voice_enabled'] = str(v)
        config_parser['Assistant']['wake_words'] = w; config_parser['StyleTTS']['volume_boost'] = str(sboost)
        config_parser['SocialStream']['session_id'] = sid; config_parser['SocialStream']['api_url'] = api
        config_parser['SocialStream']['enabled'] = str(son)
        
        with open(INI_FILE, 'w', encoding='utf-8') as f: config_parser.write(f)
        current_conf = load_settings() 
        if not k_on: 
            LOCAL_MODEL = None
            flush_vram()
        init_voice_engine()
        return "✅ Settings Applied Successfully!"

    save_btn.click(save_logic,[model_in, temp_in, penalty_in, offline_in, prompt_in, knowledge_on, engine_in, stts_url_in, stts_boost_in, ref_in, steps_in, ref_text_in, voice_on, wake_in, ssn_sid, ssn_url, ssn_on], status_msg)
    train_btn.click(teach_gem, doc_input, train_status)
    flush_btn.click(flush_vram, None, status_msg)
    
    # CLICK HANDLER FOR STYLETTS2 BAT SCRIPT
    start_stts_btn.click(launch_styletts2, None, status_msg)

if __name__ == "__main__":
    init_voice_engine()
    threading.Thread(target=run_quart, daemon=True).start()
    demo.launch(server_port=7860, quiet=True)