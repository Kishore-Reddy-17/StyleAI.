# 🎨 StyleAI — Complete Setup Guide

## Project Structure
```
styleai/
├── app.py                  ← Flask backend (main file)
├── requirements.txt        ← Python dependencies
├── .env                    ← Your API key goes here
├── templates/
│   └── index.html          ← Frontend UI
└── static/
    ├── style.css           ← Styling
    └── script.js           ← Frontend logic
```

---

## STEP 1 — Get Your FREE Groq API Key

1. Go to 👉 **https://console.groq.com**
2. Click **"Sign Up"** (it's completely free)
3. Once logged in, click **"API Keys"** in the left sidebar
4. Click **"Create API Key"** → give it a name like "StyleAI"
5. **Copy the key** (it starts with `gsk_...`)

---

## STEP 2 — Add the API Key to .env

Open the `.env` file and replace the placeholder:

```
GROQ_API_KEY=gsk_YourActualKeyHere
```

⚠️ **Don't share this key with anyone!**

---

## STEP 3 — Install Python & Dependencies

Make sure you have **Python 3.8+** installed.

Open your terminal / command prompt in the `styleai/` folder and run:

```bash
pip install -r requirements.txt
```

This installs: Flask, Groq, OpenCV, Pillow, NumPy, etc.

---

## STEP 4 — Run the App

```bash
python app.py
```

You should see:
```
============================================================
🎨  STYLE AI — Personal Fashion Advisor
============================================================
✅  Groq AI: Connected
🌐  Server: http://127.0.0.1:5000
============================================================
```

---

## STEP 5 — Open in Browser

Go to 👉 **http://127.0.0.1:5000**

---

## How to Use

1. **Select Gender** — Male or Female
2. **Upload Photo** — Drag & drop or click to browse
   - Use a clear, well-lit face photo
   - Supported: PNG, JPG, JPEG, GIF, WEBP (max 10MB)
3. **Click "Analyze My Style"**
4. Wait ~5 seconds for AI analysis
5. View your **complete style profile**:
   - Skin tone with RGB + HEX values
   - Full outfit recommendations
   - Colour palette
   - Hairstyle suggestions
   - Accessories
   - Shopping links for 7 stores

---

## Shopping Stores Supported

| Store | Category |
|-------|----------|
| Amazon.in | Everything |
| Myntra | Fashion |
| Zara | Premium Fashion |
| Flipkart | Everything |
| Puma | Footwear & Sportswear |
| Adidas | Footwear & Sportswear |
| Armani Exchange | Luxury Accessories |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Groq not connected | Check your `.env` file has correct API key |
| Face not detected | Use a clearer, well-lit frontal photo |
| Port 5000 in use | Change port in `app.py`: `port=5001` |
| OpenCV error on Mac | Run `pip install opencv-python-headless` |

---

## Tech Stack

- **Backend**: Python + Flask
- **AI Model**: Groq LLaMA 3.3 70B Versatile
- **Image Processing**: OpenCV (face detection) + Pillow
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Fonts**: Playfair Display + DM Sans

---

## Configuration (app.py)

| Parameter | Value |
|-----------|-------|
| Max File Size | 10MB |
| Groq Model | llama-3.3-70b-versatile |
| Max Tokens | 1200 |
| Temperature | 0.7 |
| Skin Tone Categories | Fair, Medium, Olive, Deep |
