import streamlit as st
import os
import zipfile
import moviepy.editor as mp
from PIL import Image
import shutil
import mysql.connector
import requests
import base64
import logging
import subprocess
import imghdr
from reverie_sdk import ReverieClient
import re
import moviepy.config as mp_config
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import sounddevice as sd
import numpy as np
import wave
import google.generativeai as genai
import tempfile
import uuid

# === PAGE CONFIG ===
st.set_page_config(layout="wide")

# === FUNCTION TO SET BACKGROUND IMAGE WITH DARK OVERLAY ===
def set_bg_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            encoded = base64.b64encode(img_bytes).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0, 0, 0, 0.65), 
                rgba(0, 0, 0, 0.65)
            ), url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.9);
        }}
        .stMarkdown, label, .stTextInput > label, .stTextArea > label {{
            color: #e0e0e0 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        }}
        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.95);
            color: #000;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
        }}
        .stFileUploader {{
            background-color: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 15px;
            font-weight: bold;
        }}
        .stSelectbox > div {{
            background-color: rgba(255,255,255,0.95) !important;
            color: #000000 !important;
        }}
        .stButton button {{
            background-color: #FF4081;
            color: #ffffff;
            padding: 12px 26px;
            border: none;
            border-radius: 12px;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 4px 4px 10px rgba(0,0,0,0.4);
        }}
        .stButton button:hover {{
            background-color: #F50057;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        logger.error(f"Background image not found at {image_path}")

# === SET BACKGROUND ===
set_bg_image("C:\\Users\\ADARSH KUMAR\\Downloads\\Ai_background.jpg")

# Set ImageMagick path
mp_config.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for FFmpeg
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    logger.info("FFmpeg is installed and accessible.")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    logger.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg.")
    st.error("FFmpeg is required but not found. Please install FFmpeg and add it to your PATH.")
    st.stop()

# Folders for file handling
UPLOAD_FOLDER = "uploads"
EXTRACT_FOLDER = "extracted_files"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"
for folder in [UPLOAD_FOLDER, EXTRACT_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# MySQL Database Configuration
db_config = {
    'user': 'root',
    'password': 'Hamzavsu777',
    'host': 'localhost',
    'database': 'picstory_db'
}

# API Configuration
PRIMARY_GEMINI_API_KEY = "AIzaSyBnrZ6cb6whzoOfCZ0XHEgjzkO555ELCAM"
BACKUP_GEMINI_API_KEY = "AIzaSyAh2GZHOXBWeYisLsUXBn5ZI--bipXy63g"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1"
GEMINI_MODEL = "gemini-1.5-flash"

# Configure Gemini API
genai.configure(api_key=PRIMARY_GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Reverie client
reverie_client = ReverieClient(
    api_key="586ee9a23c1ff03d671a827feec74dad7b357564",
    app_id="dev.2023preethi_verma",
)

# Supported languages
SUPPORTED_LANGUAGES = {
    "1": {"code": "hi-IN", "name": "Hindi", "rev_code": "hi", "speakers": ["hi_male", "hi_female"]},
    "2": {"code": "bn-IN", "name": "Bengali", "rev_code": "bn", "speakers": ["bn_male", "bn_female"]},
    "3": {"code": "kn-IN", "name": "Kannada", "rev_code": "kn", "speakers": ["kn_male", "kn_female"]},
    "4": {"code": "ml-IN", "name": "Malayalam", "rev_code": "ml", "speakers": ["ml_male", "ml_female"]},
    "5": {"code": "ta-IN", "name": "Tamil", "rev_code": "ta", "speakers": ["ta_male", "ta_female"]},
    "6": {"code": "te-IN", "name": "Telugu", "rev_code": "te", "speakers": ["te_male", "te_female"]},
    "7": {"code": "gu-IN", "name": "Gujarati", "rev_code": "gu", "speakers": ["gu_male", "gu_female"]},
    "8": {"code": "or-IN", "name": "Odia", "rev_code": "or", "speakers": ["or_male", "or_female"]},
    "9": {"code": "as-IN", "name": "Assamese", "rev_code": "as", "speakers": ["as_male", "as_female"]},
    "10": {"code": "mr-IN", "name": "Marathi", "rev_code": "mr", "speakers": ["mr_male", "mr_female"]},
    "11": {"code": "pa-IN", "name": "Punjabi", "rev_code": "pa", "speakers": ["pa_male", "pa_female"]},
    "12": {"code": "en-IN", "name": "Indian English", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "13": {"code": "kok-IN", "name": "Konkani", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "14": {"code": "doi-IN", "name": "Dogri", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "15": {"code": "brx-IN", "name": "Bodo", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "16": {"code": "ur-IN", "name": "Urdu", "rev_code": "ur", "speakers": ["ur_male", "ur_female"]},
    "17": {"code": "ks-IN", "name": "Kashmiri", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "18": {"code": "sd-IN", "name": "Sindhi", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "19": {"code": "mai-IN", "name": "Maithili", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "20": {"code": "mni-IN", "name": "Manipuri", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "21": {"code": "sa-IN", "name": "Sanskrit", "rev_code": "en", "speakers": ["en_male", "en_female"]},
    "22": {"code": "ne-IN", "name": "Nepali", "rev_code": "ne", "speakers": ["ne_male", "ne_female"]},
    "23": {"code": "sat-IN", "name": "Santali", "rev_code": "en", "speakers": ["en_male", "en_female"]},
}

# Database connection
db_conn = None
db_cursor = None

def init_db():
    global db_conn, db_cursor
    try:
        db_conn = mysql.connector.connect(**db_config)
        db_cursor = db_conn.cursor()
        db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS stories (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                user_text TEXT NOT NULL,
                detected_lang VARCHAR(10),
                output_lang VARCHAR(10),
                video_path VARCHAR(255),
                status VARCHAR(20) DEFAULT 'published',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        db_conn.commit()
        logger.info("Database connection established and tables created.")
        st.success("✅ Database connection established and tables created.")
    except mysql.connector.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

def register(username, password):
    global db_conn, db_cursor
    try:
        db_cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        db_conn.commit()
        return True
    except mysql.connector.Error as e:
        logger.error(f"Registration failed: {str(e)}")
        return False

def login(username, password):
    global db_conn, db_cursor
    try:
        db_cursor.execute("SELECT id FROM users WHERE username = %s AND password = %s", (username, password))
        user = db_cursor.fetchone()
        return user[0] if user else None
    except mysql.connector.Error as e:
        logger.error(f"Login failed: {str(e)}")
        return None

@st.cache_data
def process_input(text_input, media_paths=None):
    if not text_input or text_input == "Enter The Description":
        text_input = "A journey through breathtaking landscapes during a memorable trip."
    detected_lang = "en"
    gemini_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    api_keys = [PRIMARY_GEMINI_API_KEY, BACKUP_GEMINI_API_KEY]
    current_key_idx = 0
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"Detect the language of the following text and return the language code (e.g., 'en', 'hi'): {text_input}"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
        while current_key_idx < len(api_keys):
            try:
                response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                response.raise_for_status()
                detected_lang = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "en")
                if detected_lang not in [lang["rev_code"] for lang in SUPPORTED_LANGUAGES.values()]:
                    detected_lang = "en"
                logger.info(f"Detected language from text with key {api_keys[current_key_idx]}: {detected_lang}")
                break
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]}. Switching to backup...")
                    current_key_idx += 1
                    if current_key_idx < len(api_keys):
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All API keys exhausted: {str(e)}")
                        break
                else:
                    logger.error(f"Failed to detect language with key {api_keys[current_key_idx]}: {str(e)}")
                    break
            except requests.RequestException as e:
                logger.error(f"Failed to detect language with key {api_keys[current_key_idx]}: {str(e)}")
                break
        else:
            logger.warning("Defaulting to English due to API failures.")
    except Exception as e:
        logger.error(f"Unexpected error in process_input: {str(e)}")
        logger.warning("Defaulting to English.")
    if media_paths:
        for media_path in media_paths:
            try:
                with open(media_path, "rb") as file:
                    file_content = base64.b64encode(file.read()).decode("utf-8")
                mime_type = "image/jpeg" if media_path.lower().endswith(('.jpg', '.jpeg', '.png')) else "video/mp4"
                payload = {
                    "contents": [{"parts": [{"inline_data": {"mime_type": mime_type, "data": file_content}}, {"text": "Detect the language of any text in this media. Return the language code (e.g., 'hi', 'en')."}]}],
                    "generationConfig": {"maxOutputTokens": 10}
                }
                current_key_idx = 0
                while current_key_idx < len(api_keys):
                    try:
                        response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                        response.raise_for_status()
                        media_detected_lang = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "en")
                        if media_detected_lang not in [lang["rev_code"] for lang in SUPPORTED_LANGUAGES.values()]:
                            media_detected_lang = "en"
                        logger.info(f"Detected language from media {media_path} with key {api_keys[current_key_idx]}: {media_detected_lang}")
                        if media_detected_lang != "en":
                            detected_lang = media_detected_lang
                            break
                        break
                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 429:
                            logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]} for media {media_path}. Switching to backup...")
                            current_key_idx += 1
                            if current_key_idx < len(api_keys):
                                time.sleep(1)
                                continue
                            else:
                                logger.error(f"All API keys exhausted for media {media_path}: {str(e)}")
                                break
                        else:
                            logger.error(f"Failed to detect language from media {media_path} with key {api_keys[current_key_idx]}: {str(e)}")
                            break
                    except requests.RequestException as e:
                        logger.error(f"Failed to detect language from media {media_path} with key {api_keys[current_key_idx]}: {str(e)}")
                        break
            except Exception as e:
                logger.error(f"Unexpected error processing media {media_path}: {str(e)}")
    return text_input, detected_lang

@st.cache_data
def analyze_single_image(image_path, mime_type="image/jpeg"):
    gemini_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    api_keys = [PRIMARY_GEMINI_API_KEY, BACKUP_GEMINI_API_KEY]
    current_key_idx = 0
    headers = {"Content-Type": "application/json"}
    try:
        with open(image_path, "rb") as file:
            file_content = base64.b64encode(file.read()).decode("utf-8")
        payload = {
            "contents": [{"parts": [{"inline_data": {"mime_type": mime_type, "data": file_content}}, {"text": "Describe this image in a short sentence in English."}]}],
            "generationConfig": {"maxOutputTokens": 50}
        }
        while current_key_idx < len(api_keys):
            try:
                response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                response.raise_for_status()
                description = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                logger.info(f"Image description for {image_path} with key {api_keys[current_key_idx]}: {description}")
                time.sleep(1)
                return description
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]} for image {image_path}. Switching to backup...")
                    current_key_idx += 1
                    if current_key_idx < len(api_keys):
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All API keys exhausted for image {image_path}: {str(e)}")
                        break
                else:
                    logger.error(f"Image analysis failed for {image_path} with key {api_keys[current_key_idx]}: {str(e)}")
                    break
            except requests.RequestException as e:
                logger.error(f"Image analysis failed for {image_path} with key {api_keys[current_key_idx]}: {str(e)}")
                break
        return "Failed to analyze image."
    except Exception as e:
        logger.error(f"Unexpected error analyzing image {image_path}: {str(e)}")
        return "Failed to analyze image."

@st.cache_data
def analyze_single_video(video_path):
    gemini_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    api_keys = [PRIMARY_GEMINI_API_KEY, BACKUP_GEMINI_API_KEY]
    current_key_idx = 0
    headers = {"Content-Type": "application/json"}
    try:
        clip = mp.VideoFileClip(video_path)
        duration = clip.duration
        frame_interval = max(1, duration / 3)
        frame_paths = []
        for t in [frame_interval * i for i in range(min(3, int(duration // frame_interval) + 1))]:
            frame_path = os.path.join(TEMP_FOLDER, f"frame_{os.path.basename(video_path)}_{t}.jpg")
            clip.save_frame(frame_path, t=t)
            frame_paths.append(frame_path)
        clip.close()
        frame_descriptions = []
        for frame_path in frame_paths:
            with open(frame_path, "rb") as file:
                file_content = base64.b64encode(file.read()).decode("utf-8")
            payload = {
                "contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": file_content}}, {"text": "Describe this frame from a video in a short sentence in English."}]}],
                "generationConfig": {"maxOutputTokens": 50}
            }
            current_key_idx = 0
            while current_key_idx < len(api_keys):
                try:
                    response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                    response.raise_for_status()
                    desc = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    frame_descriptions.append(desc)
                    logger.info(f"Frame description for {frame_path} with key {api_keys[current_key_idx]}: {desc}")
                    time.sleep(1)
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:
                        logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]} for video frame {frame_path}. Switching to backup...")
                        current_key_idx += 1
                        if current_key_idx < len(api_keys):
                            time.sleep(1)
                            continue
                        else:
                            logger.error(f"All API keys exhausted for video frame {frame_path}: {str(e)}")
                            break
                    else:
                        logger.error(f"Failed to analyze frame {frame_path} with key {api_keys[current_key_idx]}: {str(e)}")
                        break
                except requests.RequestException as e:
                    logger.error(f"Failed to analyze frame {frame_path} with key {api_keys[current_key_idx]}: {str(e)}")
                    break
        combined_desc = " ".join(frame_descriptions)
        logger.info(f"Video description for {video_path}: {combined_desc}")
        return combined_desc
    except Exception as e:
        logger.error(f"Video analysis failed for {video_path}: {str(e)}")
        return "Failed to analyze video."

@st.cache_data
def generate_continuous_story(media_descriptions, user_description, output_language_code, detected_lang, num_segments):
    gemini_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    api_keys = [PRIMARY_GEMINI_API_KEY, BACKUP_GEMINI_API_KEY]
    current_key_idx = 0
    headers = {"Content-Type": "application/json"}
    target_lang = output_language_code.split('-')[0]
    media_desc_text = "\n".join([f"Media {i+1}: {desc}" for i, desc in enumerate(media_descriptions)])
    prompt = f"Based on the following user description: '{user_description}', and the descriptions of {num_segments} media items (images or videos):\n{media_desc_text}\nGenerate a continuous story in {target_lang} that flows naturally across the media, describing a journey. The story should be cohesive, reflecting the specific details of each media item in sequence, and should include a narrative arc with a beginning, middle, and end. Split the story into exactly {num_segments} non-empty parts, each corresponding to one media item. Each part should be a concise sentence of 12-16 words, suitable for narration within 6 seconds at a normal speaking pace (120-150 words per minute). Ensure each part builds on the previous part and maintains the context of a journey. Return the parts as a list separated by newlines, without any numbering or labels."
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 500}}
    for attempt in range(3):
        while current_key_idx < len(api_keys):
            try:
                response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                response.raise_for_status()
                story_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                story_segments = [segment.strip() for segment in story_text.split("\n") if segment.strip()]
                cleaned_segments = [re.sub(r'^\d+\.\s*|\d+\s*', '', segment).strip() for segment in story_segments if re.sub(r'^\d+\.\s*|\d+\s*', '', segment).strip()]
                if len(cleaned_segments) != num_segments:
                    logger.warning(f"Expected {num_segments} segments, got {len(cleaned_segments)}. Retrying...")
                    continue
                logger.info(f"Generated story segments with key {api_keys[current_key_idx]}: {cleaned_segments}")
                time.sleep(1)
                return cleaned_segments
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]} (attempt {attempt + 1}). Switching to backup...")
                    current_key_idx += 1
                    if current_key_idx < len(api_keys):
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All API keys exhausted (attempt {attempt + 1}): {str(e)}")
                        break
                else:
                    logger.error(f"Story generation failed with key {api_keys[current_key_idx]} (attempt {attempt + 1}): {str(e)}")
                    break
            except requests.RequestException as e:
                logger.error(f"Story generation failed with key {api_keys[current_key_idx]} (attempt {attempt + 1}): {str(e)}")
                break
        if current_key_idx >= len(api_keys):
            if attempt == 2:
                st.error("Failed to generate story after multiple attempts with all API keys. Please try again.")
                return None
            current_key_idx = 0
    story_segments = [f"In {target_lang}, we continued our journey, enjoying the scenery of media {i+1}." for i in range(num_segments)]
    logger.warning("Using fallback story segments due to repeated failures.")
    return story_segments

def translate_text(text, source_lang, target_lang):
    gemini_url = f"{GEMINI_BASE_URL}/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
    api_keys = [PRIMARY_GEMINI_API_KEY, BACKUP_GEMINI_API_KEY]
    current_key_idx = 0
    headers = {"Content-Type": "application/json"}
    prompt = f"Translate the following text from {source_lang} to {target_lang}: '{text}'"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 100}}
    try:
        while current_key_idx < len(api_keys):
            try:
                response = requests.post(f"{gemini_url}?key={api_keys[current_key_idx]}", json=payload, headers=headers)
                response.raise_for_status()
                translated_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", text)
                logger.info(f"Translated text from {source_lang} to {target_lang} with key {api_keys[current_key_idx]}: {translated_text}")
                time.sleep(1)
                return translated_text
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning(f"Rate limit exceeded with API key {api_keys[current_key_idx]}. Switching to backup...")
                    current_key_idx += 1
                    if current_key_idx < len(api_keys):
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All API keys exhausted: {str(e)}")
                        break
                else:
                    logger.error(f"Translation failed with key {api_keys[current_key_idx]}: {str(e)}")
                    break
            except requests.RequestException as e:
                logger.error(f"Translation failed with key {api_keys[current_key_idx]}: {str(e)}")
                break
        st.error(f"Translation failed with all API keys. Using original text as fallback.")
        return text
    except Exception as e:
        logger.error(f"Unexpected error in translate_text: {str(e)}")
        st.error(f"Translation failed unexpectedly. Using original text as fallback.")
        return text

def generate_tts_audio(text, output_path, language_code, voice="female"):
    lang_info = next((info for info in SUPPORTED_LANGUAGES.values() if info["code"] == language_code), None)
    rev_lang = lang_info["rev_code"]
    speaker = f"{rev_lang}_{voice}"
    if speaker not in lang_info["speakers"]:
        speaker = lang_info["speakers"][0]
    try:
        resp = reverie_client.tts.tts(text=text, speaker=speaker)
        resp.save_audio(output_path, create_parents=True, overwrite_existing=True)
        logger.info(f"Generated audio at {output_path} in {rev_lang} with {speaker}")
        return True
    except Exception as e:
        logger.error(f"TTS Error for {output_path}: {str(e)}")
        st.error(f"Failed to generate audio narration for {output_path}: {str(e)}")
        return False

def create_video_snippet(media_path, audio_path, output_path, text, duration=6):
    try:
        if not os.path.exists(media_path):
            logger.error(f"Media file does not exist: {media_path}")
            return False
        if media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(media_path)
                img.verify()
                img.close()
            except Exception as e:
                logger.error(f"Invalid image file {media_path}: {str(e)}")
                return False
            media_clip = mp.ImageClip(media_path).set_duration(duration).resize((854, 480))
        elif media_path.lower().endswith('.mp4'):
            media_clip = mp.VideoFileClip(media_path)
            media_clip = media_clip.subclip(0, min(duration, media_clip.duration)).resize((854, 480))
        else:
            logger.error(f"Unsupported media type: {media_path}")
            return False

        if os.path.exists(audio_path):
            audio = mp.AudioFileClip(audio_path)
            final_duration = min(duration, audio.duration, media_clip.duration if media_path.lower().endswith('.mp4') else duration)
            audio = audio.subclip(0, final_duration)
            logger.info(f"Loaded audio for {output_path} with duration {final_duration}")
        else:
            logger.warning(f"Audio file {audio_path} not found for {output_path}. Video will be silent.")
            audio = None
            final_duration = duration

        media_clip = media_clip.set_duration(final_duration)
        logger.info(f"Loaded media: {media_path}, duration: {final_duration}")

        try:
            text = text[:100]
            txt_clip = mp.TextClip(
                text,
                fontsize=20,
                color='white',
                bg_color='black',
                size=(854, 80),
                method='caption'
            )
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(final_duration)
            logger.info(f"Created text overlay for {output_path}")
        except Exception as e:
            logger.error(f"Failed to create text overlay for {output_path}: {str(e)}")
            if "ImageMagick" in str(e):
                st.error("ImageMagick is not installed or configured correctly. Text overlays will be skipped.")
            txt_clip = None

        if txt_clip and audio:
            video = mp.CompositeVideoClip([media_clip, txt_clip]).set_audio(audio)
        elif audio:
            video = media_clip.set_audio(audio)
        elif txt_clip:
            video = mp.CompositeVideoClip([media_clip, txt_clip])
        else:
            video = media_clip

        video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            verbose=True,
            logger=None
        )
        logger.info(f"Created video snippet at {output_path}, has audio: {video.audio is not None if 'video' in locals() else 'N/A'}")
        if not os.path.exists(output_path):
            logger.error(f"Video file {output_path} was not created.")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to create video snippet at {output_path}: {str(e)}")
        st.error(f"Failed to create video snippet at {output_path}: {str(e)}")
        return False

def concatenate_videos(video_paths, output_path, background_music_path=None):
    try:
        clips = []
        for video_path in video_paths:
            if os.path.exists(video_path):
                clip = mp.VideoFileClip(video_path)
                clips.append(clip)
            else:
                logger.warning(f"Video file {video_path} does not exist, skipping.")
        if not clips:
            logger.error("No valid video clips to concatenate.")
            st.error("No valid video clips available to concatenate.")
            return False
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        if background_music_path and os.path.exists(background_music_path):
            bg_audio = mp.AudioFileClip(background_music_path).set_duration(final_clip.duration).volumex(0.3)
            final_audio = mp.CompositeAudioClip([final_clip.audio, bg_audio])
            final_clip = final_clip.set_audio(final_audio)
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=True,
            logger=None
        )
        for clip in clips:
            clip.close()
        final_clip.close()
        logger.info(f"Concatenated video saved at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to concatenate videos: {str(e)}")
        st.error(f"Failed to concatenate videos: {str(e)}")
        return False

def record_audio(filename, stop_event: threading.Event, samplerate=44100):
    audio_data = []
    
    def callback(indata, frames, time, status):
        if not stop_event.is_set():
            audio_data.append(indata.copy())
        else:
            raise sd.StopStream()

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16, callback=callback):
            st.info("🎙 Recording started...")
            while not stop_event.is_set():
                time.sleep(0.1)
        st.success("✅ Recording completed.")
        audio_data = np.concatenate(audio_data, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
    except sd.StopStream:
        pass

def get_gemini_response(input_msg, audio_path, mime_type="audio/wav"):
    with open(audio_path, "rb") as f:
        audio = genai.upload_file(f, mime_type=mime_type)
    response = model.generate_content([audio, input_msg])
    return response.text

def process_snippet(args):
    i, media_path, segment, output_lang_code, voice = args
    audio_path = os.path.join(TEMP_FOLDER, f"audio_segment_{i}.mp3")
    video_path = os.path.join(TEMP_FOLDER, f"video_segment_{i}.mp4")
    logger.info(f"Processing snippet {i+1} with media: {media_path}, segment: {segment}")
    if not os.path.exists(media_path):
        logger.error(f"Media file does not exist: {media_path}")
        return None
    if generate_tts_audio(segment, audio_path, output_lang_code, voice):
        logger.info(f"Generated audio: {audio_path}, exists: {os.path.exists(audio_path)}")
        if create_video_snippet(media_path, audio_path, video_path, segment):
            logger.info(f"Generated video: {video_path}, exists: {os.path.exists(video_path)}")
            return video_path
    logger.error(f"Failed to process snippet {i+1} at {media_path}")
    return None

def process_files(media_paths, user_description, output_lang_code, voice="female"):
    user_description, detected_lang = process_input(user_description, media_paths)
    media_descriptions = []
    for media_path in media_paths:
        with st.spinner(f"Analyzing {os.path.basename(media_path)}..."):
            if media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                desc = analyze_single_image(media_path)
            elif media_path.lower().endswith('.mp4'):
                desc = analyze_single_video(media_path)
            else:
                desc = "Unsupported media type."
            media_descriptions.append(desc)
    st.write("### Media Descriptions:")
    for i, desc in enumerate(media_descriptions):
        st.write(f"Media {i+1}: {desc}")
    with st.spinner("Generating story..."):
        story_segments = generate_continuous_story(media_descriptions, user_description, output_lang_code, detected_lang, len(media_paths))
    if not story_segments:
        st.error("Failed to generate story.")
        return None, None, None, None, None
    return media_paths, story_segments, user_description, detected_lang, output_lang_code

def cleanup_temp():
    for folder in [TEMP_FOLDER, EXTRACT_FOLDER, UPLOAD_FOLDER]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                os.makedirs(folder)
            except PermissionError as e:
                logger.error(f"Permission denied while deleting {folder}: {str(e)}. Retrying after delay...")
                time.sleep(2)
                try:
                    shutil.rmtree(folder)
                    os.makedirs(folder)
                except Exception as e2:
                    logger.error(f"Failed to delete {folder} after retry: {str(e2)}")
                    st.warning(f"Could not clean up {folder} due to file lock. Please close any applications using the files and try again.")

def main():
    global db_conn, db_cursor
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = False
    if 'user_description' not in st.session_state:
        st.session_state.user_description = "Enter The Description"
    if 'media_paths' not in st.session_state:
        st.session_state.media_paths = None
    if 'story_segments' not in st.session_state:
        st.session_state.story_segments = None
    if 'english_segments' not in st.session_state:
        st.session_state.english_segments = None
    if 'user_desc' not in st.session_state:
        st.session_state.user_desc = None
    if 'det_lang' not in st.session_state:
        st.session_state.det_lang = None
    if 'out_lang' not in st.session_state:
        st.session_state.out_lang = None
    if 'output_lang_code' not in st.session_state:
        st.session_state.output_lang_code = None
    if 'voice' not in st.session_state:
        st.session_state.voice = None
    if 'edited_segments' not in st.session_state:
        st.session_state.edited_segments = None
    if 'edited_english_segments' not in st.session_state:
        st.session_state.edited_english_segments = None
    if 'show_complete_story' not in st.session_state:
        st.session_state.show_complete_story = False
    if 'video_snippets' not in st.session_state:
        st.session_state.video_snippets = None
    if 'selected_snippets' not in st.session_state:
        st.session_state.selected_snippets = None
    if 'show_edit_section' not in st.session_state:
        st.session_state.show_edit_section = False

    st.sidebar.markdown("<h3 style='font-size:24px;'>⚙ User Options</h3>", unsafe_allow_html=True)

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            st.markdown("<h3 style='font-size:28px;'>🔑 Login</h3>", unsafe_allow_html=True)
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                user_id = login(login_username, login_password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with tab2:
            st.markdown("<h3 style='font-size:28px;'>📝 Register</h3>", unsafe_allow_html=True)
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            if st.button("Register"):
                if register(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists or registration failed.")
    else:
        st.markdown("<h1 style='font-size:44px; font-weight: bold;'>📸 PicStory: Generate Stories from Images and Videos</h1>", unsafe_allow_html=True)
        
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("<h4 style='font-size:22px;'>📁 Upload a ZIP file containing images (.jpg, .jpeg, .png) or videos (Max 10 img 0r video) (.mp4)</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop file here", type=["zip"], help="Limit 200MB per file • ZIP")

        st.markdown("<h3 style='font-size:28px;'>📝 Story Description</h3>", unsafe_allow_html=True)
        cols = st.columns([2, 1])
        with cols[0]:
            user_description = st.text_area(
                " ",
                placeholder="Enter The Description",
                value=st.session_state.user_description,
                key="user_desc_input",
                help="You can type a description here or use the microphone to record one."
            )
            st.session_state.user_description = user_description
        with cols[1]:
            mic_lang_choice = st.selectbox("🎙 Mic Language:", [f"{lang['name']} ({lang['code']})" for lang in SUPPORTED_LANGUAGES.values()], index=0)
            mic_lang_code = SUPPORTED_LANGUAGES[[key for key, val in SUPPORTED_LANGUAGES.items() if f"{val['name']} ({val['code']})" == mic_lang_choice][0]]["code"]

            record_label = "🎤 Record Audio" if not st.session_state.recording_state else "🎤 Stop Recording"
            if st.button(record_label):
                if not st.session_state.recording_state:
                    st.session_state.recording_state = True
                    stop_event = threading.Event()
                    st.session_state.stop_event = stop_event

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_path = temp_file.name

                    threading.Thread(target=record_audio, args=(temp_path, stop_event), daemon=True).start()
                    st.session_state.temp_path = temp_path
                else:
                    st.session_state.recording_state = False
                    st.session_state.stop_event.set()

                    if 'temp_path' in st.session_state:
                        st.info("Transcribing...")
                        result = get_gemini_response(f"Transcribe this audio in {mic_lang_code}", st.session_state.temp_path)
                        if result:
                            st.session_state.user_description = result.strip()
                            st.success("Transcription complete!")
                            st.write(f"Transcribed Description: {st.session_state.user_description}")
                        else:
                            st.error("Failed to transcribe audio.")
                        os.unlink(st.session_state.temp_path)
                        del st.session_state.temp_path
                    st.rerun()

        language_choice = st.selectbox("Select output language:", [f"{lang['name']} ({lang['code']})" for lang in SUPPORTED_LANGUAGES.values()], index=0)
        output_lang_code = SUPPORTED_LANGUAGES[[key for key, val in SUPPORTED_LANGUAGES.items() if f"{val['name']} ({val['code']})" == language_choice][0]]["code"]
        voice = st.selectbox("Select voice:", ["male", "female"])
        background_music = st.file_uploader("Upload background music (optional, MP3)", type=["mp3"])

        if st.button("Generate Story"):
            if not uploaded_file:
                st.error("Please upload a ZIP file to proceed.")
                return

            if not uploaded_file.name.endswith('.zip'):
                st.error("Only ZIP files are accepted. Please upload a ZIP file containing images (.jpg, .jpeg, .png) or videos (.mp4).")
                return

            with st.spinner("Processing files and generating story..."):
                cleanup_temp()
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                media_paths = []
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_FOLDER)
                for root, _, files in os.walk(EXTRACT_FOLDER):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4')):
                            if len(media_paths) < 10:
                                media_paths.append(file_path)
                                logger.info(f"Found media: {file_path}")
                            else:
                                logger.info(f"Ignoring additional file {file_path} due to 10-file limit.")
                        else:
                            logger.info(f"Ignoring unsupported file: {file}")

                num_images = len([p for p in media_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))])
                num_videos = len([p for p in media_paths if p.lower().endswith('.mp4')])
                st.info(f"Processed {num_images + num_videos} files: {num_images} images, {num_videos} videos")

                if not media_paths:
                    st.error("No valid images or videos found in the ZIP file. Please upload a ZIP with supported files.")
                    return

                media_paths, story_segments, user_desc, det_lang, out_lang = process_files(media_paths, st.session_state.user_description, output_lang_code, voice)
                if not story_segments:
                    st.error("Failed to generate story. Please try again.")
                    return

                st.session_state.media_paths = media_paths
                st.session_state.story_segments = story_segments
                st.session_state.user_desc = user_desc
                st.session_state.det_lang = det_lang
                st.session_state.out_lang = out_lang
                st.session_state.output_lang_code = output_lang_code
                st.session_state.voice = voice
                st.session_state.edited_segments = story_segments.copy()
                st.session_state.edited_english_segments = None
                st.session_state.english_segments = None
                st.session_state.show_complete_story = False
                st.session_state.show_edit_section = False
                st.session_state.video_snippets = None
                st.session_state.selected_snippets = None

        if st.session_state.story_segments:
            target_lang = st.session_state.output_lang_code.split('-')[0]
            
            if st.session_state.english_segments is None:
                st.session_state.english_segments = []
                for segment in st.session_state.story_segments:
                    with st.spinner(f"Translating segment to English for review..."):
                        segment_in_english = translate_text(segment, target_lang, "en") if target_lang != "en" else segment
                    st.session_state.english_segments.append(segment_in_english)

            st.markdown("<h3 style='font-size:28px;'>📜 AI-Generated Story Segments</h3>", unsafe_allow_html=True)
            for i, (segment, english_segment) in enumerate(zip(st.session_state.story_segments, st.session_state.english_segments)):
                st.write(f"Segment {i+1} (in {target_lang}): {segment}")
                st.write(f"Segment {i+1} (in English): {english_segment}")
            
            if st.button("Edit Story Segments"):
                st.session_state.show_edit_section = True
                st.session_state.edited_english_segments = st.session_state.english_segments.copy()
                st.rerun()

        if st.session_state.show_edit_section and st.session_state.story_segments:
            st.markdown("<h3 style='font-size:28px;'>✏ Edit Story Segments (in English)</h3>", unsafe_allow_html=True)
            target_lang = st.session_state.output_lang_code.split('-')[0]
            
            edited_english_segments = []
            for i, (original_segment, english_segment) in enumerate(zip(st.session_state.story_segments, st.session_state.edited_english_segments)):
                st.write(f"Original Segment {i+1} (in {target_lang}): {original_segment}")
                st.write(f"Original English Translation: {english_segment}")
                edited_segment = st.text_input(
                    f"Edit Segment {i+1} (in English):",
                    value=english_segment,
                    key=f"edit_segment_{i}",
                    help="Edit this segment in English. It will be translated back to the target language."
                )
                edited_english_segments.append(edited_segment)
            
            st.session_state.edited_english_segments = edited_english_segments

            if st.button("Generate Edited Story"):
                with st.spinner("Translating edited segments to target language..."):
                    edited_segments = []
                    for i, edited_segment in enumerate(st.session_state.edited_english_segments):
                        translated_segment = translate_text(edited_segment, "en", target_lang) if target_lang != "en" else edited_segment
                        edited_segments.append(translated_segment)
                    st.session_state.edited_segments = edited_segments
                    st.session_state.show_edit_section = False
                    st.rerun()

        if st.session_state.edited_segments and not st.session_state.show_edit_section:
            st.markdown("<h3 style='font-size:28px;'>📜 Edited Story Segments (in Target Language)</h3>", unsafe_allow_html=True)
            for i, segment in enumerate(st.session_state.edited_segments):
                st.write(f"Segment {i+1}: {segment}")

            if st.button("View Complete Edited Story"):
                st.session_state.show_complete_story = True
                st.rerun()

        if st.session_state.show_complete_story and st.session_state.edited_segments:
            st.markdown("<h3 style='font-size:28px;'>📖 Complete Edited Story</h3>", unsafe_allow_html=True)
            complete_story = " ".join(st.session_state.edited_segments)
            st.write(complete_story)

            if st.button("Create Video Snippets"):
                if not st.session_state.media_paths or not st.session_state.edited_segments:
                    st.error("Required data is missing. Please generate the story again.")
                    return

                with st.spinner("Generating video snippets..."):
                    video_snippets = []
                    with ThreadPoolExecutor() as executor:
                        video_snippets = list(executor.map(
                            process_snippet,
                            [(i, media, seg, st.session_state.output_lang_code, st.session_state.voice)
                             for i, (media, seg) in enumerate(zip(st.session_state.media_paths, st.session_state.edited_segments))]
                        ))
                    video_snippets = [v for v in video_snippets if v]

                    if not video_snippets:
                        st.error("Failed to generate video snippets. Please check the logs for details.")
                        return

                    st.session_state.video_snippets = video_snippets
                    st.session_state.selected_snippets = list(range(len(video_snippets)))

        if st.session_state.video_snippets:
            st.markdown("<h3 style='font-size:28px;'>🎥 Preview Video Snippets</h3>", unsafe_allow_html=True)
            for i, video_path in enumerate(st.session_state.video_snippets):
                if i in st.session_state.selected_snippets:
                    st.write(f"Snippet {i+1}:")
                    with open(video_path, "rb") as video_file:
                        st.video(video_file, format="video/mp4")
                    if st.button(f"Remove Snippet {i+1}", key=f"remove_{i}"):
                        st.session_state.selected_snippets.remove(i)
                        st.rerun()

            if st.session_state.selected_snippets:
                if st.button("Concatenate Selected Snippets"):
                    with st.spinner("Concatenating selected snippets..."):
                        final_snippets = [st.session_state.video_snippets[i] for i in st.session_state.selected_snippets]
                        final_video_path = os.path.join(OUTPUT_FOLDER, "final_story.mp4")
                        bg_music_path = os.path.join(UPLOAD_FOLDER, background_music.name) if background_music else None
                        if bg_music_path:
                            with open(bg_music_path, "wb") as f:
                                f.write(background_music.getbuffer())
                            if not os.path.exists(bg_music_path):
                                st.error(f"Background music file {bg_music_path} was not saved correctly.")
                        success = concatenate_videos(final_snippets, final_video_path, bg_music_path)
                        if success:
                            st.markdown("<h3 style='font-size:28px;'>🎬 Final Concatenated Video:</h3>", unsafe_allow_html=True)
                            with open(final_video_path, "rb") as final_video:
                                st.video(final_video, format="video/mp4")
                            with open(final_video_path, "rb") as file:
                                st.download_button(
                                    label="Download Video",
                                    data=file,
                                    file_name="final_story.mp4",
                                    mime="video/mp4"
                                )
                            time.sleep(2)
                            try:
                                db_cursor.executemany(
                                    'INSERT INTO stories (user_id, user_text, detected_lang, output_lang, video_path) VALUES (%s, %s, %s, %s, %s)',
                                    [(st.session_state.user_id, st.session_state.user_desc, st.session_state.det_lang, st.session_state.out_lang, video_path)
                                     for video_path in final_snippets]
                                )
                                db_conn.commit()
                                st.success(f"Saved {len(final_snippets)} story segments to database.")
                            except mysql.connector.Error as e:
                                logger.error(f"Failed to save to database: {e}")
                                st.error(f"Failed to save to database: {e}")
                            cleanup_temp()
                        else:
                            st.error("Failed to concatenate videos. Please check the logs for details.")
            else:
                st.warning("No snippets selected. Please keep at least one snippet to concatenate.")

if __name__ == "__main__":
    main()
