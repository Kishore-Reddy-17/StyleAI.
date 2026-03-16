from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import os
import base64
from datetime import datetime
from dotenv import load_dotenv
import io
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# ─── Auth & DB Config ────────────────────────
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'styleai-secret-change-me-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///styleai_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'auth_page'
login_manager.login_message = 'Please log in to access StyleAI.'
login_manager.login_message_category = 'error'


# ─── User Model ──────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses_count = db.Column(db.Integer, default=0)
    # Profile fields
    profile_photo = db.Column(db.Text, nullable=True)   # base64 data-URI
    gender = db.Column(db.String(10), default='male')
    height = db.Column(db.String(20), nullable=True)
    preferred_brands = db.Column(db.String(255), nullable=True)
    budget_min = db.Column(db.Integer, default=500)
    budget_max = db.Column(db.Integer, default=5000)
    city = db.Column(db.String(100), nullable=True)     # for weather
    profile_complete = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─── Create tables if they don't exist ───────
with app.app_context():
    db.create_all()


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Force browser to always fetch fresh static files (no caching during dev)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def no_cache(response):
    if 'static' in request.path:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ─────────────────────────────────────────────
# Hugging Face Inference Setup
# ─────────────────────────────────────────────
hf_client = None
HF_ERROR = None

try:
    from huggingface_hub import InferenceClient
    api_key = os.getenv('HF_TOKEN')
    if api_key and api_key != 'your_hf_token_here':
        hf_client = InferenceClient(api_key=api_key)
        print("✅ HuggingFace Inference Client initialized successfully")
    else:
        HF_ERROR = "No valid HF_TOKEN found in .env file"
        print(f"⚠️  {HF_ERROR}")
except Exception as e:
    HF_ERROR = f"HuggingFace import failed: {e}"
    print(f"❌ {HF_ERROR}")

# ─────────────────────────────────────────────
# YOLO Face Detector Setup
# ─────────────────────────────────────────────
yolo_face = None
import torch

try:
    from ultralytics import YOLO
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True   # auto-tune cuDNN kernels for speed
    # Try HuggingFace face-specific model first, then general YOLO
    for _model_id in ["hf://hustvl/yolov8n-face.pt", "yolov8n.pt"]:
        try:
            yolo_face = YOLO(_model_id)
            yolo_face.to(_device)
            print(f"✅ YOLO face detector loaded: {_model_id} on {_device.upper()}")
            break
        except Exception as _me:
            print(f"   ↳ Could not load {_model_id}: {_me}")
    if yolo_face is None:
        print("⚠️  YOLO unavailable — will use Haar cascade for face detection")
except Exception as e:
    print(f"⚠️  YOLO import failed ({e}) — will use Haar cascade fallback")


# ─────────────────────────────────────────────
# DINOv2 Vision Model Setup
# ─────────────────────────────────────────────
dino_processor = None
dino_model = None

try:
    from transformers import AutoImageProcessor, AutoModel

    # Try giant first (~4 GB fp16), fall back to base (~350 MB) for smaller GPUs
    for _dino_name in ["facebook/dinov2-giant", "facebook/dinov2-base"]:
        try:
            print(f"   Loading DINOv2: {_dino_name} …")
            dino_processor = AutoImageProcessor.from_pretrained(_dino_name)
            dino_model = AutoModel.from_pretrained(
                _dino_name,
                dtype=torch.float16
            ).to(_device)
            dino_model.eval()
            print(f"✅ DINOv2 vision model loaded: {_dino_name} on {_device.upper()}")
            break
        except Exception as _de:
            print(f"   ↳ Could not load {_dino_name}: {_de}")
            dino_processor = None
            dino_model = None

    if dino_model is None:
        print("⚠️  DINOv2 unavailable — vision features will not be extracted")
except Exception as e:
    print(f"⚠️  DINOv2 import failed ({e})")


# ─────────────────────────────────────────────
# SkinToneHead — linear classifier on DINOv2 features
# ─────────────────────────────────────────────
import torch.nn as nn

# ── 9-tone Monk Skin Tone Scale (MST) ──────────────────────
# Covers the full human complexion range from Ivory to Ebony
SKIN_TONE_CLASSES = [
    "Ivory",        # MST-1 — porcelain, very fair, cool/pink
    "Fair",         # MST-2 — fair, light beige, neutral
    "Light Beige",  # MST-3 — light-medium, warm beige
    "Sandy Beige",  # MST-4 — medium-light, golden warmth
    "Golden Beige", # MST-5 — true medium, warm gold
    "Warm Tan",     # MST-6 — medium-deep, warm tan
    "Olive Brown",  # MST-7 — rich olive/brown
    "Caramel",      # MST-8 — deep, warm caramel brown
    "Ebony",        # MST-9 — very deep, rich dark
]

# Each class maps to itself — all 9 tones fed directly to styling
SKIN_CLASS_TO_TONE = {t: t for t in SKIN_TONE_CLASSES}

skin_head = None
try:
    if dino_model is not None:
        class SkinToneHead(nn.Module):
            def __init__(self, hidden_dim, num_classes=9):  # 9-tone MST scale
                super().__init__()
                self.fc = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                return self.fc(x)

        skin_head = SkinToneHead(
            hidden_dim=dino_model.config.hidden_size,
            num_classes=len(SKIN_TONE_CLASSES)
        ).to(_device).half()   # fp16 — matches DINOv2 output dtype
        skin_head.eval()
        print(f"✅ SkinToneHead ready (input_dim={dino_model.config.hidden_size}, classes={len(SKIN_TONE_CLASSES)})")
    else:
        print("⚠️  SkinToneHead skipped — DINOv2 not loaded")
except Exception as e:
    print(f"⚠️  SkinToneHead init failed: {e}")


# ─────────────────────────────────────────────
# DINOv2 Feature Extraction Helper
# ─────────────────────────────────────────────
def get_dino_features(pil_image):
    """
    Extract DINOv2 CLS-token embedding from a PIL image.
    Returns a float16 tensor of shape (1, hidden_dim) or None if unavailable.
    Use for: similarity search, gender classification, style matching, etc.
    """
    if dino_model is None or dino_processor is None:
        return None
    try:
        with torch.no_grad():
            inputs = dino_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            # Cast pixel values to fp16 to match model precision
            inputs["pixel_values"] = inputs["pixel_values"].half()
            outputs = dino_model(**inputs)
            # CLS token — shape: (1, hidden_dim)
            return outputs.last_hidden_state[:, 0, :]
    except Exception as e:
        print(f"DINOv2 feature extraction error: {e}")
        return None


# ─────────────────────────────────────────────
# predict_skin_tone_deep — YOLO + DINOv2 + SkinToneHead
# ─────────────────────────────────────────────
def predict_skin_tone_deep(img_rgb_array):
    """
    Full deep-learning skin tone pipeline:
      1. YOLO  → detect face bounding boxes
      2. DINOv2 → extract CLS embedding per face
      3. SkinToneHead → classify into 6 skin tone classes

    Args:
        img_rgb_array: np.ndarray in RGB, HxWx3

    Returns:
        List of dicts: [{"bbox": [x1,y1,x2,y2], "skin_class": str, "skin_tone": str, "confidence": float}]
        Returns [] if any model is unavailable or no face is found.
    """
    if yolo_face is None or dino_model is None or skin_head is None:
        return []

    try:
        from PIL import Image as PILImage

        # Step 1: YOLO face detection
        results = yolo_face(img_rgb_array, verbose=False, conf=0.4)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        outputs = []
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # Clamp to image bounds
            h, w = img_rgb_array.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = img_rgb_array[y1:y2, x1:x2]
            face_pil = PILImage.fromarray(face_crop)

            # Step 2: DINOv2 feature extraction
            with torch.no_grad():
                inputs = dino_processor(images=face_pil, return_tensors="pt")
                inputs = {k: v.to(_device) for k, v in inputs.items()}
                inputs["pixel_values"] = inputs["pixel_values"].half()
                features = dino_model(**inputs).last_hidden_state[:, 0, :]  # (1, 1536)

                # Step 3: SkinToneHead classification
                skin_logits = skin_head(features)                    # (1, 6)
                probs = torch.softmax(skin_logits.float(), dim=1)    # float32 for softmax
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()

            skin_class = SKIN_TONE_CLASSES[pred_idx]
            skin_tone  = SKIN_CLASS_TO_TONE[skin_class]

            outputs.append({
                "bbox":       [x1, y1, x2, y2],
                "skin_class": skin_class,   # granular (6-class)
                "skin_tone":  skin_tone,    # styling bucket (4-class)
                "confidence": round(confidence, 4),
            })
            print(f"🧠 DINOv2 prediction: {skin_class} → {skin_tone} (conf={confidence:.2%})")

        return outputs

    except Exception as e:
        print(f"predict_skin_tone_deep error: {e}")
        return []


# ─────────────────────────────────────────────
# Skin Tone Detection (OpenCV fallback)
# ─────────────────────────────────────────────
def detect_skin_tone(image_data):
    """
    Detect skin tone from image bytes using YOLO (primary) or Haar cascade (fallback).
    Returns (tone_name, r, g, b, hex_color)
    """
    try:
        import cv2
        from PIL import Image

        # Load image
        img_pil = Image.open(io.BytesIO(image_data))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_array = np.array(img_pil)
        img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # ══ PRIMARY PATH: YOLO → DINOv2 → SkinToneHead ══
        deep_results = predict_skin_tone_deep(img_array)
        deep_tone = None
        if deep_results:
            # Use the highest-confidence face prediction
            best = max(deep_results, key=lambda x: x["confidence"])
            deep_tone = best["skin_tone"]   # 4-bucket name for styling

        # ══ ALWAYS: OpenCV YCrCb for actual RGB colour value ══
        face_roi = None

        # Try YOLO for face crop (colour ROI)
        if yolo_face is not None:
            try:
                results = yolo_face(img_array, verbose=False, conf=0.4)
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy.cpu().numpy()]
                    best_box = boxes.xyxy.cpu().numpy()[int(np.argmax(areas))].astype(int)
                    x1, y1, x2, y2 = best_box[0], best_box[1], best_box[2], best_box[3]
                    fw, fh = x2-x1, y2-y1
                    face_roi = img_bgr[y1+int(fh*0.1):y1+int(fh*0.6),
                                       x1+int(fw*0.2):x1+int(fw*0.8)]
            except Exception:
                pass

        # Haar cascade fallback for colour ROI
        if face_roi is None:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = img_bgr[y+int(h*0.1):y+int(h*0.6), x+int(w*0.2):x+int(w*0.8)]

        if face_roi is None or face_roi.size == 0:
            raise ValueError("No face detected in the image.")

        # YCrCb skin pixel extraction for colour
        roi_ycrcb  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        skin_mask  = cv2.inRange(roi_ycrcb,
                                 np.array([0,133,77],  dtype=np.uint8),
                                 np.array([255,173,127], dtype=np.uint8))
        skin_pixels = face_roi[skin_mask > 0]
        if len(skin_pixels) < 100:
            skin_pixels = face_roi.reshape(-1, 3)

        avg_bgr = np.mean(skin_pixels, axis=0)
        b, g, r = int(avg_bgr[0]), int(avg_bgr[1]), int(avg_bgr[2])
        brightness = r*0.299 + g*0.587 + b*0.114

        # ── Tone name: prefer deep-learning result, fallback to brightness thresholds ──
        if deep_tone:
            tone = deep_tone
            print(f"✅ Tone from DINOv2: {tone}")
        else:
            # 9-bucket brightness ladder (perceptual luminance via rec.601)
            if   brightness > 215: tone = "Ivory"
            elif brightness > 195: tone = "Fair"
            elif brightness > 178: tone = "Light Beige"
            elif brightness > 158: tone = "Sandy Beige"
            elif brightness > 138: tone = "Golden Beige"
            elif brightness > 118: tone = "Warm Tan"
            elif brightness > 96:  tone = "Olive Brown"
            elif brightness > 72:  tone = "Caramel"
            else:                  tone = "Ebony"
            print(f"✅ Tone from OpenCV brightness: {tone} ({brightness:.1f})")

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"🎨 Final — {tone} | RGB({r},{g},{b}) | HEX {hex_color}")
        return tone, r, g, b, hex_color

    except Exception as e:
        print(f"Skin tone detection error: {e}")
        raise e


# ─────────────────────────────────────────────
# Gender Detection (DeepFace) — pre-import at startup
# ─────────────────────────────────────────────
_deepface = None
try:
    from deepface import DeepFace as _DeepFace
    _deepface = _DeepFace
    print("✅ DeepFace loaded for gender detection")
except Exception as _dfe:
    print(f"⚠️  DeepFace unavailable ({_dfe}) — gender validation disabled")


def detect_gender_from_image(image_data):
    """
    Detect gender from image bytes using DeepFace.
    Returns (dominant_gender, man_conf, woman_conf)
    dominant_gender is 'Man', 'Woman', or None on failure.
    """
    if _deepface is None:
        print("⚠️  DeepFace not loaded — skipping gender check")
        return None, 0, 0
    try:
        from PIL import Image as PILImage
        img_pil = PILImage.open(io.BytesIO(image_data))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_array = np.array(img_pil)

        result = _deepface.analyze(
            img_array,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='opencv',   # fastest, most reliable on Windows
            silent=True
        )
        if isinstance(result, list):
            result = result[0]

        dominant_gender = result.get('dominant_gender')   # 'Man' or 'Woman'
        gender_conf     = result.get('gender', {})
        man_conf        = gender_conf.get('Man', 0)
        woman_conf      = gender_conf.get('Woman', 0)
        face_conf       = result.get('face_confidence', 0)
        print(f"👤 DeepFace → {dominant_gender}  Man:{man_conf:.1f}%  Woman:{woman_conf:.1f}%  face_conf:{face_conf:.2f}")
        return dominant_gender, man_conf, woman_conf

    except Exception as e:
        print(f"Gender detection error: {e}")
        return None, 0, 0


# ─────────────────────────────────────────────
# HuggingFace Styling Recommendations
# ─────────────────────────────────────────────
def get_styling_recommendations(skin_tone, gender, r, g, b, pref_height='', pref_budget='', pref_brands='', pref_occasion='casual', pref_weather=''):
    """Call Hugging Face Inference API to get personalized styling recommendations."""

    if hf_client is None:
        return generate_fallback_recommendations(skin_tone, gender)

    gender_label = "male" if gender.lower() == "male" else "female"
    gender_items = (
        "shirts, trousers/chinos, shoes" if gender_label == "male"
        else "tops/blouses, skirts/trousers, shoes"
    )

    pref_section = "CLIENT CONTEXT FOR THIS RECOMMENDATION:\n"
    if pref_height:
        pref_section += f"- Height: {pref_height} (Categorize the client by height and choose flattering silhouettes based on current fashion trends.)\n"
    if pref_budget:
        pref_section += f"- Budget Range (INR): {pref_budget} (Select brands and products strictly within this price range.)\n"
    if pref_brands:
        pref_section += f"- Preferred Brands: {pref_brands} (Prioritize these brands in recommendations.)\n"
    if pref_occasion:
        pref_section += f"- Occasion: {pref_occasion} (Design the entire outfit specifically for this occasion.)\n"
    if pref_weather:
        pref_section += f"- Current Weather: {pref_weather} (Choose fabrics, layers, and colors appropriate for these weather conditions.)\n"

    prompt = f"""You are a world-class personal fashion stylist. A {gender_label} client has a {skin_tone} skin tone (RGB: {r},{g},{b}). This is measured on a 9-point Monk Skin Tone Scale, where {skin_tone} represents their specific complexion.

{pref_section}
Provide a detailed, structured styling recommendation in EXACTLY this format (use → as bullet prefix):

DRESS_CODE
→ Formal
→ Business
→ Casual
→ Party

SUGGESTED_OUTFIT
→ [Write 2-3 sentence description of a complete outfit ({gender_items}) that complements their skin tone perfectly. If height was provided, explain how this outfit flatters their height category based on current trends.]

SHIRT_DETAILS
→ Color: [specific color]
→ Type: [type e.g. Oxford, Polo, Linen, Silk blouse]
→ Brand: [premium brand available in India, respecting the budget/brands preference if given]
→ Fabric: [fabric type]

PANT_DETAILS
→ Color: [specific color]
→ Type: [e.g. Slim fit chinos, Wide leg trousers, A-line skirt]
→ Brand: [premium brand available in India, respecting the budget/brands preference if given]
→ Fabric: [fabric type]

SHOES_DETAILS
→ Color: [specific color]
→ Type: [e.g. Loafers, Oxford, Heels, Sneakers]
→ Brand: [premium brand available in India, respecting the budget/brands preference if given]

HAIRSTYLE
→ Style: [specific hairstyle name]
→ How-to: [2-3 sentence detailed styling guide and maintenance tips]

ACCESSORIES
→ 1. [Accessory with description]
→ 2. [Accessory with description]
→ 3. [Accessory with description]
→ 4. [Accessory with description]

COLOR_PALETTE
→ Primary: [color name]
→ Secondary: [color name]
→ Accent: [color name]

WHY_IT_WORKS
→ [2-3 sentences explaining exactly why these choices complement {skin_tone} skin tone, referencing colour theory and undertones. If height was provided, also justify the silhouette choices.]

Be specific, confident, and fashion-forward. The outfit MUST perfectly suit the occasion ('{pref_occasion}') and be appropriate for the current weather ({pref_weather or 'typical Indian conditions'}). All recommendations must be appropriate for the Indian market."""


    try:
        response = hf_client.chat_completion(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a professional fashion stylist AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return generate_fallback_recommendations(skin_tone, gender)


def generate_fallback_recommendations(skin_tone, gender):
    """Offline fallback recommendations when Groq is unavailable."""
    # 9-tone Monk scale — expanded palette per tone
    palettes = {
        "Ivory":        ("Soft Lavender",  "Blush Pink",    "Silver",     "#d4b8d0"),
        "Fair":         ("Navy Blue",       "Soft Pink",     "Pearl White","#9b7fa6"),
        "Light Beige":  ("Sky Blue",        "Peach",         "Mint Green", "#87ceeb"),
        "Sandy Beige":  ("Terracotta",      "Ivory",         "Gold",       "#c8843c"),
        "Golden Beige": ("Olive Green",     "Camel",         "Burnt Orange","#8b7355"),
        "Warm Tan":     ("Mustard Yellow",  "Rust",          "Tan",         "#c49a3a"),
        "Olive Brown":  ("Burgundy",        "Forest Green",  "Copper",     "#8b4513"),
        "Caramel":      ("Royal Blue",      "Emerald Green", "Bright Gold","#1a3f6f"),
        "Ebony":        ("Cobalt Blue",     "Bright Orange", "Pure White", "#002366"),
    }
    p, s, a, _ = palettes.get(skin_tone, palettes["Golden Beige"])

    if gender.lower() == "male":
        outfit = f"A {p.lower()} slim-fit linen shirt paired with beige chinos and tan loafers — effortlessly smart for your {skin_tone} complexion."
        shirt = f"Color: {p}\n→ Type: Slim Fit Linen\n→ Brand: Mango Man\n→ Fabric: Linen"
        pant = f"Color: Beige\n→ Type: Slim Chinos\n→ Brand: Zara\n→ Fabric: Cotton"
        shoes = f"Color: Tan\n→ Type: Loafers\n→ Brand: Clarks"
        hair = "Style: Classic Taper\n→ How-to: Ask your barber for a taper fade on sides; keep top 2–3 inches. Use a matte clay to style."
        acc = "1. Minimalist silver watch\n→ 2. Leather belt (tan)\n→ 3. Stud earring\n→ 4. Aviator sunglasses"
    else:
        outfit = f"A {p.lower()} wrap dress or a flowy blouse with high-waisted trousers — a timeless combination for your {skin_tone} skin tone."
        shirt = f"Color: {p}\n→ Type: Wrap Blouse\n→ Brand: Zara\n→ Fabric: Chiffon"
        pant = f"Color: Ivory\n→ Type: High-waist Wide Leg\n→ Brand: Mango\n→ Fabric: Crepe"
        shoes = f"Color: Nude\n→ Type: Block Heel Mules\n→ Brand: Steve Madden"
        hair = "Style: Loose Beach Waves\n→ How-to: Apply sea-salt spray to damp hair, scrunch, and diffuse. Finish with a light serum for shine."
        acc = "1. Gold hoop earrings\n→ 2. Layered delicate necklace\n→ 3. Woven tote bag\n→ 4. Stack bangles"

    return f"""DRESS_CODE
→ Formal
→ Business
→ Casual
→ Party

SUGGESTED_OUTFIT
→ {outfit}

SHIRT_DETAILS
→ {shirt}

PANT_DETAILS
→ {pant}

SHOES_DETAILS
→ {shoes}

HAIRSTYLE
→ {hair}

ACCESSORIES
→ {acc}

COLOR_PALETTE
→ Primary: {p}
→ Secondary: {s}
→ Accent: {a}

WHY_IT_WORKS
→ {skin_tone} skin tones are beautifully complemented by {p.lower()} and {s.lower()} hues, which create a harmonious contrast that enhances your natural complexion. The {a.lower()} accent adds warmth and dimension, making your overall look vibrant and polished."""


# ─────────────────────────────────────────────
# Shopping Links Generator
# ─────────────────────────────────────────────
def get_shopping_links(skin_tone, gender):
    """Return curated shopping links based on skin tone and gender."""

    gender_q = "men" if gender.lower() == "male" else "women"
    # 9-tone Monk scale → shirt/pant/shoe colour mapping
    tone_map = {
        "Ivory":        {"shirt_color": "pale blue",      "pant_color": "light grey",  "shoe_color": "white"},
        "Fair":         {"shirt_color": "navy blue",       "pant_color": "grey",         "shoe_color": "white"},
        "Light Beige":  {"shirt_color": "sky blue",        "pant_color": "beige",        "shoe_color": "nude"},
        "Sandy Beige":  {"shirt_color": "terracotta",      "pant_color": "cream",        "shoe_color": "tan"},
        "Golden Beige": {"shirt_color": "olive green",     "pant_color": "khaki",        "shoe_color": "camel"},
        "Warm Tan":     {"shirt_color": "mustard yellow",  "pant_color": "brown",        "shoe_color": "cognac"},
        "Olive Brown":  {"shirt_color": "burgundy",        "pant_color": "dark navy",    "shoe_color": "chocolate brown"},
        "Caramel":      {"shirt_color": "royal blue",      "pant_color": "charcoal",     "shoe_color": "white"},
        "Ebony":        {"shirt_color": "cobalt blue",     "pant_color": "black",        "shoe_color": "bright white"},
    }
    colors = tone_map.get(skin_tone, tone_map["Golden Beige"])
    sc, pc, shc = colors["shirt_color"], colors["pant_color"], colors["shoe_color"]

    def encode(q): return q.replace(" ", "+")

    products = [
        {
            "name": f"{sc.title()} Shirt / Top",
            "stores": [
                {"name": "Amazon.in", "icon": "🛒", "url": f"https://www.amazon.in/s?k={encode(gender_q+'+'+sc+'+shirt')}", "color": "#ff9900"},
                {"name": "Myntra", "icon": "👗", "url": f"https://www.myntra.com/{gender_q}-shirts?rawQuery={encode(sc+'+shirt')}", "color": "#ff3f6c"},
                {"name": "Zara", "icon": "✦", "url": f"https://www.zara.com/in/en/search?searchTerm={encode(sc+'+shirt')}", "color": "#000000"},
            ]
        },
        {
            "name": f"{pc.title()} Trousers / Bottom",
            "stores": [
                {"name": "Myntra", "icon": "👗", "url": f"https://www.myntra.com/{gender_q}-trousers?rawQuery={encode(pc+'+trouser')}", "color": "#ff3f6c"},
                {"name": "Flipkart", "icon": "🏪", "url": f"https://www.flipkart.com/search?q={encode(gender_q+'+'+pc+'+trouser')}", "color": "#2874f0"},
                {"name": "Zara", "icon": "✦", "url": f"https://www.zara.com/in/en/search?searchTerm={encode(pc+'+trouser')}", "color": "#000000"},
            ]
        },
        {
            "name": f"{shc.title()} Footwear",
            "stores": [
                {"name": "Amazon.in", "icon": "🛒", "url": f"https://www.amazon.in/s?k={encode(gender_q+'+'+shc+'+shoes')}", "color": "#ff9900"},
                {"name": "Myntra", "icon": "👗", "url": f"https://www.myntra.com/{gender_q}-shoes?rawQuery={encode(shc+'+shoes')}", "color": "#ff3f6c"},
                {"name": "Puma", "icon": "🐆", "url": f"https://in.puma.com/in/en/search?q={encode(shc+'+shoes')}", "color": "#e31837"},
                {"name": "Adidas", "icon": "⚡", "url": f"https://www.adidas.co.in/search?q={encode(shc+'+shoes')}", "color": "#000000"},
            ]
        },
        {
            "name": "Premium Accessories",
            "stores": [
                {"name": "Amazon.in", "icon": "🛒", "url": f"https://www.amazon.in/s?k={encode(gender_q+'+fashion+accessories')}", "color": "#ff9900"},
                {"name": "Myntra", "icon": "👗", "url": f"https://www.myntra.com/{gender_q}-accessories", "color": "#ff3f6c"},
                {"name": "Armani Exchange", "icon": "💎", "url": f"https://www.armaniexchange.com/en/catalog/category/view/id/117?q={encode(gender_q)}", "color": "#8b6914"},
            ]
        },
        {
            "name": "Complete Outfit Sets",
            "stores": [
                {"name": "Zara", "icon": "✦", "url": f"https://www.zara.com/in/en/{gender_q}/", "color": "#000000"},
                {"name": "Myntra", "icon": "👗", "url": f"https://www.myntra.com/", "color": "#ff3f6c"},
                {"name": "Armani Exchange", "icon": "💎", "url": "https://www.armaniexchange.com/en/", "color": "#8b6914"},
            ]
        },
    ]
    return products


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/proxy-image')
def proxy_image():
    """Proxy fashion images server-side. Prioritizes Bing Images for exact product matches."""
    query = request.args.get('q', 'fashion+outfit')
    w = request.args.get('w', '400')
    h = request.args.get('h', '300')
    seed = abs(hash(query)) % 1000

    urls_to_try = []

    # 1. Bing Image Search for exact product images (bypasses DDGS rate limits)
    try:
        import urllib.parse
        import re
        search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
        req_bing = urllib.request.Request(search_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        html = urllib.request.urlopen(req_bing, timeout=5).read().decode('utf-8', errors='ignore')
        # Bing stores image URLs in murl&quot;:&quot;http...&quot;
        img_matches = re.findall(r'murl&quot;:&quot;(https?://.*?\.(?:jpg|jpeg|png|webp))&quot;', html, re.IGNORECASE)
        # Add the top 3 results to our connection queue
        urls_to_try.extend(img_matches[:3])
    except Exception as e:
        print(f"Bing Scraper error for '{query}': {e}")

    # Process URLs
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
            })
            with urllib.request.urlopen(req, timeout=4) as resp:
                img_data = resp.read()
                content_type = resp.headers.get('Content-Type', 'image/jpeg')
                # Make sure we didn't get an HTML error page back
                if img_data and len(img_data) > 1000 and 'html' not in content_type.lower():
                    response = Response(img_data, mimetype=content_type)
                    response.headers['Cache-Control'] = 'public, max-age=3600'
                    return response
        except Exception:
            continue

    # SVG gradient fallback — always works, looks premium
    colors = ['#1a1814', '#2c3e7a', '#c9a96e', '#252320']
    c1, c2 = colors[seed % len(colors)], colors[(seed + 2) % len(colors)]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
      <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:{c1}"/>
        <stop offset="100%" style="stop-color:{c2}"/>
      </linearGradient></defs>
      <rect width="{w}" height="{h}" fill="url(#g)"/>
      <text x="50%" y="50%" font-family="Georgia,serif" font-size="18" fill="#c9a96e"
        text-anchor="middle" dominant-baseline="middle" opacity="0.6">✦ StyleAI</text>
    </svg>'''
    return Response(svg, mimetype='image/svg+xml')



@app.route('/detect-live', methods=['POST'])
def detect_live():
    """Lightweight endpoint for real-time webcam skin tone detection."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        image_bytes = base64.b64decode(data['image'])
        tone, r, g, b, hex_color = detect_skin_tone(image_bytes)
        return jsonify({'success': True, 'skin_tone': {'name': tone, 'r': r, 'g': g, 'b': b, 'hex': hex_color}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)


@app.route('/auth')
def auth_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth.html', active_tab='login')


@app.route('/auth/login', methods=['POST'])
def auth_login():
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user, remember=True)
        flash(f'Welcome back, {user.name}! 👋', 'success')
        next_page = request.args.get('next')
        return redirect(next_page or url_for('index'))

    flash('Invalid email or password. Please try again.', 'error')
    return render_template('auth.html', active_tab='login')


@app.route('/auth/register', methods=['POST'])
def auth_register():
    name     = request.form.get('name', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    terms    = request.form.get('terms')

    if not terms:
        flash('You must accept the Terms of Service to register.', 'error')
        return render_template('auth.html', active_tab='register')

    if len(password) < 8:
        flash('Password must be at least 8 characters long.', 'error')
        return render_template('auth.html', active_tab='register')

    if User.query.filter_by(email=email).first():
        flash('An account with that email already exists. Please log in.', 'error')
        return render_template('auth.html', active_tab='login')

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=True)
    # New users go to profile setup first
    return redirect(url_for('profile_setup'))


@app.route('/auth/logout')
@login_required
def auth_logout():
    logout_user()
    flash('You have been signed out. See you soon!', 'success')
    return redirect(url_for('auth_page'))


# ─────────────────────────────────────────────
# Profile Setup Routes
# ─────────────────────────────────────────────

@app.route('/profile/setup')
@login_required
def profile_setup():
    return render_template('profile.html', user=current_user)


@app.route('/profile/save', methods=['POST'])
@login_required
def profile_save():
    try:
        current_user.gender = request.form.get('gender', 'male')
        current_user.height = request.form.get('height', '')
        current_user.preferred_brands = request.form.get('preferred_brands', '')
        current_user.budget_min = int(request.form.get('budget_min', 500) or 500)
        current_user.budget_max = int(request.form.get('budget_max', 5000) or 5000)
        current_user.city = request.form.get('city', '')

        # Handle profile photo
        photo_data = request.form.get('profile_photo_data', '')
        if photo_data and photo_data.startswith('data:image'):
            current_user.profile_photo = photo_data

        current_user.profile_complete = True
        db.session.commit()
        flash(f'Profile saved! Welcome to StyleAI, {current_user.name} ✦', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/profile/update-photo', methods=['POST'])
@login_required
def update_photo():
    try:
        data = request.get_json()
        photo_data = data.get('photo_data', '')
        if photo_data and photo_data.startswith('data:image'):
            current_user.profile_photo = photo_data
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid photo data'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────
# Weather API Route
# ─────────────────────────────────────────────

@app.route('/weather')
@login_required
def get_weather():
    """Fetch current weather for user's city using Open-Meteo (no API key needed)."""
    try:
        import requests as req
        city = request.args.get('city', current_user.city or 'Mumbai')

        # Step 1: Geocode the city name to lat/lon using Open-Meteo geocoder
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={req.utils.quote(city)}&count=1&language=en&format=json"
        geo_resp = req.get(geo_url, timeout=5).json()
        if not geo_resp.get('results'):
            return jsonify({'success': False, 'error': 'City not found'}), 404

        loc = geo_resp['results'][0]
        lat, lon = loc['latitude'], loc['longitude']

        # Step 2: Fetch weather with WMO code
        weather_url = (f"https://api.open-meteo.com/v1/forecast?"
                       f"latitude={lat}&longitude={lon}"
                       f"&current=temperature_2m,relative_humidity_2m,weathercode,windspeed_10m"
                       f"&timezone=auto")
        w = req.get(weather_url, timeout=5).json()
        current = w.get('current', {})

        wmo_code = current.get('weathercode', 0)
        temp = current.get('temperature_2m', 25)
        humidity = current.get('relative_humidity_2m', 50)
        wind = current.get('windspeed_10m', 0)

        # Map WMO weather code to description + emoji
        def wmo_to_desc(code):
            if code == 0: return ('Clear sky', '☀️', 'sunny')
            elif code <= 3: return ('Partly cloudy', '⛅', 'cloudy')
            elif code <= 48: return ('Foggy', '🌫️', 'cold')
            elif code <= 57: return ('Drizzle', '🌦️', 'rainy')
            elif code <= 67: return ('Rainy', '🌧️', 'rainy')
            elif code <= 77: return ('Snow', '❄️', 'cold')
            elif code <= 82: return ('Rain showers', '🌧️', 'rainy')
            elif code <= 99: return ('Thunderstorm', '⛈️', 'stormy')
            else: return ('Unknown', '🌡️', 'mild')

        desc, emoji, condition = wmo_to_desc(wmo_code)

        # Season logic (Northern Hemisphere)
        month = datetime.now().month
        if month in [12, 1, 2]: season = 'winter'
        elif month in [3, 4, 5]: season = 'spring'
        elif month in [6, 7, 8]: season = 'summer'
        else: season = 'autumn'

        return jsonify({
            'success': True,
            'city': loc['name'],
            'country': loc.get('country', ''),
            'temperature': round(temp, 1),
            'humidity': humidity,
            'wind': round(wind, 1),
            'description': desc,
            'emoji': emoji,
            'condition': condition,
            'season': season
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────
# Public Health Check
# ─────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'hf_connected': hf_client is not None,
        'hf_error': HF_ERROR
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if this is an Avatar Builder request (JSON)
        if request.is_json:
            data = request.get_json()
            if data and data.get('is_avatar'):
                gender = data.get('gender', 'male')
                tone_idx = int(data.get('avatar_tone_index', 5))
                pref_height = data.get('pref_height', '')
                pref_budget = data.get('pref_budget', '')
                pref_brands = data.get('pref_brands', '')
                pref_occasion = data.get('pref_occasion', 'casual')
                pref_weather = data.get('pref_weather', '')
                
                # Map 1-9 index to the standard 9-class Monk Scale names and approximations
                # Indices 1-9 map to SKIN_TONE_CLASSES indices 0-8
                if 1 <= tone_idx <= 9:
                    tone = SKIN_TONE_CLASSES[tone_idx - 1]
                else:
                    tone = "Medium"
                
                # Approximate hex colors for the UI based on Monk scale
                tone_colors = {
                    1: (246, 237, 228, "#f6ede4"), 2: (243, 231, 219, "#f3e7db"), 
                    3: (247, 234, 208, "#f7ead0"), 4: (234, 218, 186, "#eadaba"), 
                    5: (215, 189, 150, "#d7bd96"), 6: (160, 126, 86,  "#a07e56"), 
                    7: (130, 92,  67,  "#825c43"), 8: (96,  65,  52,  "#604134"), 
                    9: (58,  49,  42,  "#3a312a")
                }
                r, g, b, hex_color = tone_colors.get(tone_idx, tone_colors[5])
                
                # Step 2 & 3: Get recommendations based on avatar traits
                recommendations_text = get_styling_recommendations(tone, gender, r, g, b, pref_height, pref_budget, pref_brands, pref_occasion, pref_weather)
                products = get_shopping_links(tone, gender)
                
                return jsonify({
                    'success': True,
                    'is_avatar': True,
                    'avatar_tone_index': tone_idx,
                    'skin_tone': {'name': tone, 'r': r, 'g': g, 'b': b, 'hex': hex_color},
                    'recommendations': recommendations_text,
                    'products': products,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        # --- Standard Photo Upload Flow ---
        # Validate file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({'success': False, 'error': f'File type .{ext} not allowed. Use: PNG, JPG, JPEG, GIF, WEBP'}), 400

        gender = request.form.get('gender', 'male')
        pref_height = request.form.get('pref_height', '')
        pref_budget = request.form.get('pref_budget', '')
        pref_brands = request.form.get('pref_brands', '')
        image_data = file.read()

        # ── Gender Validation ──────────────────────────────────────────
        detected_gender, man_conf, woman_conf = detect_gender_from_image(image_data)

        if detected_gender is not None:
            detected_norm = 'male' if detected_gender == 'Man' else 'female'
            selected_norm = gender.lower()

            # Block if dominant gender disagrees with selection (>50% = clear majority)
            top_conf = max(man_conf, woman_conf)
            if detected_norm != selected_norm and top_conf > 50:
                detected_label = 'Male'  if detected_norm == 'male'   else 'Female'
                selected_label = 'Male'  if selected_norm == 'male'   else 'Female'
                return jsonify({
                    'success': False,
                    'gender_mismatch': True,
                    'detected_gender': detected_label,
                    'selected_gender': selected_label,
                    'error': (
                        f'Gender mismatch detected! '
                        f'AI identified a {detected_label} face in your photo '
                        f'({detected_label} confidence: {top_conf:.0f}%), '
                        f'but you selected {selected_label}. '
                        f'Please upload a {selected_label.lower()} photo or '
                        f'change the gender selection to {detected_label}.'
                    )
                }), 422
        # ─────────────────────────────────────────────────────────────

        # Step 1: Detect skin tone
        tone, r, g, b, hex_color = detect_skin_tone(image_data)

        # Step 2: Get AI styling recommendations
        recommendations_text = get_styling_recommendations(tone, gender, r, g, b, pref_height, pref_budget, pref_brands)

        # Step 3: Get shopping links
        products = get_shopping_links(tone, gender)


        return jsonify({
            'success': True,
            'skin_tone': {
                'name': tone,
                'r': r, 'g': g, 'b': b,
                'hex': hex_color
            },
            'recommendations': recommendations_text,
            'products': products,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎨  STYLE AI — Personal Fashion Advisor")
    print("="*60)
    print(f"✅  HuggingFace AI: {'Connected' if hf_client else '⚠️  Not connected — using fallback'}")
    if HF_ERROR:
        print(f"    ↳ {HF_ERROR}")
    print(f"🌐  Server: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
