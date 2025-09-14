import os
import sys
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
from PIL import Image
import io
import uuid
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI
import ollama
import time
import requests
import subprocess
import termcolor

# Set up logging to avoid Werkzeug's default messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app) 

# Global state for the in-memory vector database
knowledge_base_embeddings = {}
knowledge_base_facts = {}
embedding_model_name = "multi-qa-mpnet-base-dot-v1"
embedding_model = None

# Providers and API keys will be set dynamically
current_provider = None
current_provider_index = 0
provider_options = []
PROVIDER_MAP = {}
CLIENTS = {}
MODEL_ALIASES = {
    'gemini': {
        'text': ['gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-1.0-pro'],
        'vision': ['gemini-2.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.0-pro-vision'],
    },
    'openai': {
        'text': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
        'vision': ['gpt-4o', 'gpt-4o-mini'],
    },
    'ollama': {
        'text': 'llama3',
        'vision': 'llava',
    }
}

def start_ollama_service():
    """Checks if Ollama service is running and starts it if not."""
    print("Checking for Ollama service... 🔍", end='\r', flush=True)
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        print("Ollama service is already running. ✅")
        return True
    except requests.exceptions.ConnectionError:
        print("Ollama service not found. Attempting to start... ⏳", end='\r', flush=True)
        try:
            subprocess.Popen(['ollama', 'serve'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
            requests.get("http://localhost:11434/api/tags", timeout=1)
            print("Ollama service started. ✅")
            return True
        except FileNotFoundError:
            print(termcolor.colored("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH. ❌", "red"))
            return False
        except Exception as e:
            print(termcolor.colored(f"Error starting Ollama service: {e} ❌", "red"))
            return False

def build_initial_knowledge_base():
    """Builds the in-memory knowledge base from the knowledge.md file."""
    print("Building knowledge base embeddings... 🧠", flush=True)
    try:
        global embedding_model
        embedding_model = SentenceTransformer(embedding_model_name)
        
        with open('knowledge.md', 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                embedding = embedding_model.encode(line)
                knowledge_base_embeddings[i] = embedding
                knowledge_base_facts[i] = line
        print(termcolor.colored(f"Knowledge base built with {len(knowledge_base_facts)} entries. ✅", "green"))
    except FileNotFoundError:
        print(termcolor.colored("Knowledge.md not found. Starting with an empty knowledge base. ⚠️", "yellow"))
    except Exception as e:
        print(termcolor.colored(f"Error building knowledge base: {e} ❌", "red"))
        sys.exit(1)

def get_provider_client(provider):
    if provider not in CLIENTS:
        if provider == 'gemini':
            CLIENTS[provider] = genai
        elif provider == 'openai':
            CLIENTS[provider] = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        elif provider == 'ollama':
            CLIENTS[provider] = ollama
    return CLIENTS.get(provider)

def get_current_model(is_vision_task):
    if is_vision_task:
        return PROVIDER_MAP.get(current_provider, {}).get('vision')
    else:
        return PROVIDER_MAP.get(current_provider, {}).get('text')

def handle_rate_limit_exceeded(exception):
    global current_provider_index, current_provider
    print(termcolor.colored(f"\nRate limit exceeded on {current_provider}. ⏳", "yellow"))
    current_provider_index = (current_provider_index + 1) % len(provider_options)
    current_provider = provider_options[current_provider_index]
    print(termcolor.colored(f"Switched to provider: {current_provider} ✅", "green"))
    raise ResourceExhausted("Rate limit exceeded and could not switch providers.")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ResourceExhausted)
)
def api_call_with_retry(payload, is_vision_task):
    global current_provider
    client = get_provider_client(current_provider)
    model_name = get_current_model(is_vision_task)
    
    if current_provider == 'gemini':
        try:
            return client.GenerativeModel(model_name).generate_content(payload).text
        except ResourceExhausted as e:
            handle_rate_limit_exceeded(e)
        except Exception as e:
            print(termcolor.colored(f"Gemini API Error: {e} ❌", "red"))
            raise e
    
    elif current_provider == 'openai':
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=payload
            ).choices[0].message.content
        except Exception as e:
            if 'rate limit' in str(e).lower():
                handle_rate_limit_exceeded(e)
            else:
                print(termcolor.colored(f"OpenAI API Error: {e} ❌", "red"))
                raise e
            
    elif current_provider == 'ollama':
        return ollama.chat(model=model_name, messages=payload)['message']['content']

    return "No valid provider found."

# --- Flask Routes and App Logic ---
def save_image_to_disk(base64_data):
    try:
        image_bytes = base64.b64decode(base64_data)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join('images', filename)
        os.makedirs('images', exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def update_knowledge_base(new_fact, image_filename=None):
    try:
        entry = f"- {new_fact}"
        if image_filename:
            entry += f" (image: {image_filename})"
        with open('knowledge.md', 'a') as f:
            f.write(f"\n{entry}")
        new_index = len(knowledge_base_facts)
        new_embedding = embedding_model.encode(entry)
        knowledge_base_embeddings[new_index] = new_embedding
        knowledge_base_facts[new_index] = entry
        print(f"New fact saved: {entry}")
        return True
    except Exception as e:
        print(f"Error updating knowledge base: {e}")
        return False

def save_new_fact(new_fact):
    return update_knowledge_base(new_fact)

def find_relevant_facts(query, top_k=3):
    if not knowledge_base_facts:
        return [], []
    query_embedding = embedding_model.encode(query)
    similarities = {
        index: cosine_similarity([query_embedding], [embedding])[0][0]
        for index, embedding in knowledge_base_embeddings.items()
    }
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    relevant_facts, relevant_images = [], []
    for index, similarity in sorted_similarities[:top_k]:
        fact = knowledge_base_facts[index]
        relevant_facts.append(fact)
        match = re.search(r'\(image: ([a-f0-9-]+\.png)\)', fact)
        if match:
            image_filename = match.group(1)
            image_path = os.path.join('images', image_filename)
            if os.path.exists(image_path):
                relevant_images.append(Image.open(image_path))
    return relevant_facts, relevant_images

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ResourceExhausted)
)
def ask_bot():
    data = request.json
    user_input = data.get('question', '')
    image_data = data.get('image', None)
    conversation_history = data.get('history', [])
    
    try:
        is_vision_task = bool(image_data)

        if image_data:
            image_filename = save_image_to_disk(image_data)
            if not image_filename:
                return jsonify({'answer': "Sorry, couldn't save image."})
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            
            prompt_text = "Analyze this image and user text. Extract the core fact only."
            payload = [{'role': 'user', 'parts': [{'text': prompt_text}, image]}]
            fact_response = api_call_with_retry(payload, is_vision_task).strip()
            
            if update_knowledge_base(fact_response, image_filename):
                return jsonify({'answer': f"Saved from image: {fact_response}"})
            return jsonify({'answer': "Failed to save image information."})

        if "save" in user_input.lower() or "remember" in user_input.lower():
            prompt = f"Extract the key fact from: '{user_input}'"
            payload = [{'role': 'user', 'parts': [{'text': prompt}]}]
            new_fact = api_call_with_retry(payload, is_vision_task).strip()
            if save_new_fact(new_fact):
                return jsonify({'answer': f"Saved: {new_fact}"})
            return jsonify({'answer': "Failed to save."})

        relevant_facts, relevant_images = find_relevant_facts(user_input)
        rag_context = "\n".join(relevant_facts)
        prompt = f"""You are a helpful and accurate personal assistant. Your sole purpose is to answer questions using only the provided context. If the answer is not in the context, state 'I don't know.' Do not use any outside knowledge.
        
        Context:
        {rag_context}
        
        Question:
        {user_input}"""
        
        payload_parts = [{'text': prompt}]
        for img in relevant_images:
            payload_parts.append(img)
        payload = conversation_history + [{'role': 'user', 'parts': payload_parts}]
        answer = api_call_with_retry(payload, is_vision_task)
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': f"Error: {e}"})

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--providers':
        provider_options = sys.argv[2:]
    else:
        print("No providers specified. Defaulting to Ollama...")
        provider_options = ['ollama']

    if not provider_options:
        print(termcolor.colored("❌ No providers specified.", "red"))
        sys.exit(1)

    current_provider = provider_options[0]

    for provider in provider_options:
        PROVIDER_MAP[provider] = {'text': None, 'vision': None}

        if provider == 'ollama':
            if not start_ollama_service():
                continue
            
            text_model = MODEL_ALIASES['ollama']['text']
            vision_model = MODEL_ALIASES['ollama']['vision']
            
            try:
                ollama.show(text_model)
                PROVIDER_MAP[provider]['text'] = text_model
                print(termcolor.colored(f"✅ Text model '{text_model}' found.", "green"))
            except Exception:
                print(termcolor.colored(f"❌ Text model '{text_model}' not found. Run 'ollama pull {text_model}'.", "red"))

            try:
                ollama.show(vision_model)
                PROVIDER_MAP[provider]['vision'] = vision_model
                print(termcolor.colored(f"✅ Vision model '{vision_model}' found.", "green"))
            except Exception:
                print(termcolor.colored(f"❌ Vision model '{vision_model}' not found. Run 'ollama pull {vision_model}'.", "red"))
        
        elif provider == 'gemini':
            if not os.environ.get('GEMINI_API_KEY'):
                print(termcolor.colored("❌ GEMINI_API_KEY not set.", "red"))
                continue
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            PROVIDER_MAP[provider]['text'] = f"models/{MODEL_ALIASES['gemini']['text'][0]}"
            PROVIDER_MAP[provider]['vision'] = f"models/{MODEL_ALIASES['gemini']['vision'][0]}"

        elif provider == 'openai':
            if not os.environ.get('OPENAI_API_KEY'):
                print(termcolor.colored("❌ OPENAI_API_KEY not set.", "red"))
                continue
            PROVIDER_MAP[provider]['text'] = MODEL_ALIASES['openai']['text'][0]
            PROVIDER_MAP[provider]['vision'] = MODEL_ALIASES['openai']['vision'][0]

    if not any(p in PROVIDER_MAP and (PROVIDER_MAP[p]['text'] is not None or PROVIDER_MAP[p]['vision'] is not None) for p in provider_options):
        print(termcolor.colored("❌ No active providers with models.", "red"))
        sys.exit(1)

    build_initial_knowledge_base()
    print(termcolor.colored("🚀 Starting app on port 8000...", "cyan"))
    app.run(host='0.0.0.0', port=8000)