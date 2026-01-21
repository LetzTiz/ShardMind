"""
ShardMind v1.4 - Archaeological Fragment Analysis & Reconstruction
"""

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import io
from PIL import Image
import base64
import pickle
import qrcode
import hashlib
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import uuid
import requests
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_VERSION = "1.4"
USERS_DB_PATH = Path("shardmind_users.pkl")
BASE_URL = "https://shardmind.streamlit.app"

# =============================================================================
# TRANSLATIONS (IMPROVED)
# =============================================================================

TRANSLATIONS = {
    'de': {
        # App
        'app_title': 'ShardMind',
        'app_subtitle': 'Archäologische Fragmentanalyse',
        'app_tagline': 'Analysiere und rekonstruiere zerbrochene Objekte',
        
        # Auth
        'login': 'Anmelden',
        'register': 'Registrieren', 
        'username': 'Benutzername',
        'password': 'Passwort',
        'password_confirm': 'Passwort bestätigen',
        'login_btn': '🔐 Anmelden',
        'register_btn': '📝 Registrieren',
        'logout_btn': '🚪',
        'login_error': 'Falsche Anmeldedaten',
        'register_success': 'Registrierung erfolgreich!',
        'register_error_exists': 'Benutzername bereits vergeben',
        'register_error_password': 'Passwörter stimmen nicht überein',
        'register_error_short': 'Passwort mind. 4 Zeichen',
        'account_info': 'Dein Account wird dauerhaft gespeichert',
        
        # Settings
        'settings': '⚙️ Einstellungen',
        'ai_settings': '🤖 KI-Einstellungen',
        'ai_provider': 'KI-Anbieter',
        'api_key': 'API-Schlüssel',
        'use_ai': 'KI für Beschreibungen nutzen',
        'ai_none': 'Keine KI',
        
        # Upload & Analysis
        'upload_photos': '📤 Fotos hochladen',
        'upload_hint': 'JPG, PNG - Scherben auf hellem Hintergrund',
        'min_size': 'Min. Fragmentgröße',
        'min_size_help': 'Kleinere Objekte werden ignoriert',
        'cluster_sens': 'Gruppierungsstärke',
        'cluster_help': 'Höher = strengere Gruppierung',
        'detection_mode': 'Erkennungsmodus',
        'mode_auto': 'Automatisch',
        'mode_light_bg': 'Heller Hintergrund',
        'mode_dark_bg': 'Dunkler Hintergrund',
        'mode_high_contrast': 'Hoher Kontrast',
        'project': 'Projekt/Grabung',
        'analyze_btn': '🔬 Analysieren',
        'clear_btn': '🗑️ Leeren',
        'analyzing': 'Analysiere Bild...',
        
        # Tabs
        'tab_start': '🏠 Start',
        'tab_gallery': '🏺 Galerie',
        'tab_groups': '📦 Gruppen',
        'tab_reconstruction': '🧩 Rekonstruktion',
        'tab_database': '💾 Datenbank',
        'tab_labels': '🏷️ Etiketten',
        'tab_help': '❓ Hilfe',
        
        # Gallery
        'detected_fragments': 'Erkannte Fragmente',
        'no_fragments': 'Keine Fragmente erkannt',
        'fragments': 'Fragmente',
        'groups': 'Gruppen',
        
        # Groups
        'groups_title': 'Gruppierung',
        'group_name': 'Gruppenname',
        'save_btn': '💾 In Datenbank speichern',
        'pieces': 'Teile',
        'no_groups': 'Keine Gruppen erkannt',
        'saved': 'Gespeichert!',
        
        # Reconstruction
        'reconstruction_title': 'Rekonstruktion',
        'select_group': 'Gruppe auswählen',
        'canvas_size': 'Arbeitsfläche',
        'calculate_btn': '🔄 Berechnen',
        'matches_found': 'Übereinstimmungen',
        'export_btn': '📥 Als Bild speichern',
        'edge_matches': 'Kantenübereinstimmungen',
        
        # Database
        'database_title': 'Meine Sammlung',
        'fragments_saved': 'Gespeicherte Fragmente',
        'groups_saved': 'Gespeicherte Gruppen',
        'db_empty': 'Noch keine Daten gespeichert',
        
        # Labels
        'labels_title': 'Etiketten erstellen',
        'label_source': 'Quelle',
        'label_session': 'Aktuelle Analyse',
        'label_database': 'Datenbank',
        'label_custom': 'Eigenes Etikett',
        'custom_label_id': 'Eigene ID (optional)',
        'custom_label_name': 'Bezeichnung',
        'custom_label_desc': 'Beschreibung',
        'custom_label_project': 'Projekt',
        'create_pdf': '📄 PDF erstellen',
        'add_custom_label': '➕ Etikett hinzufügen',
        'custom_labels_list': 'Eigene Etiketten',
        'clear_custom': 'Liste leeren',
        
        # Search
        'search_title': 'Fragment suchen',
        'search_input': 'Fragment-ID eingeben',
        'search_btn': '🔍 Suchen',
        'search_found': 'Fragment gefunden!',
        'search_not_found': 'Nicht gefunden',
        
        # Demo
        'demo_title': 'Demo ausprobieren',
        'demo_desc': 'Lade ein Testbild herunter:',
        'demo_pottery': 'Keramikscherben',
        'demo_plate': 'Tellerbruch',
        'demo_download': '📥 Herunterladen',
        'demo_hint': 'Nach dem Download: Links hochladen → Analysieren',
        
        # Help
        'help_title': 'Anleitung',
        
        # General
        'upload_first': 'Lade zuerst Fotos hoch',
        'success_fragments': 'Fragmente erkannt!',
        'language': 'Sprache',
    },
    'en': {
        # App
        'app_title': 'ShardMind',
        'app_subtitle': 'Archaeological Fragment Analysis',
        'app_tagline': 'Analyze and reconstruct broken objects',
        
        # Auth
        'login': 'Login',
        'register': 'Register',
        'username': 'Username',
        'password': 'Password',
        'password_confirm': 'Confirm Password',
        'login_btn': '🔐 Login',
        'register_btn': '📝 Register',
        'logout_btn': '🚪',
        'login_error': 'Invalid credentials',
        'register_success': 'Registration successful!',
        'register_error_exists': 'Username already taken',
        'register_error_password': 'Passwords do not match',
        'register_error_short': 'Password min. 4 characters',
        'account_info': 'Your account is permanently saved',
        
        # Settings
        'settings': '⚙️ Settings',
        'ai_settings': '🤖 AI Settings',
        'ai_provider': 'AI Provider',
        'api_key': 'API Key',
        'use_ai': 'Use AI for descriptions',
        'ai_none': 'No AI',
        
        # Upload & Analysis
        'upload_photos': '📤 Upload Photos',
        'upload_hint': 'JPG, PNG - Fragments on light background',
        'min_size': 'Min. Fragment Size',
        'min_size_help': 'Smaller objects are ignored',
        'cluster_sens': 'Grouping Strength',
        'cluster_help': 'Higher = stricter grouping',
        'detection_mode': 'Detection Mode',
        'mode_auto': 'Automatic',
        'mode_light_bg': 'Light Background',
        'mode_dark_bg': 'Dark Background',
        'mode_high_contrast': 'High Contrast',
        'project': 'Project/Excavation',
        'analyze_btn': '🔬 Analyze',
        'clear_btn': '🗑️ Clear',
        'analyzing': 'Analyzing image...',
        
        # Tabs
        'tab_start': '🏠 Start',
        'tab_gallery': '🏺 Gallery',
        'tab_groups': '📦 Groups',
        'tab_reconstruction': '🧩 Reconstruction',
        'tab_database': '💾 Database',
        'tab_labels': '🏷️ Labels',
        'tab_help': '❓ Help',
        
        # Gallery
        'detected_fragments': 'Detected Fragments',
        'no_fragments': 'No fragments detected',
        'fragments': 'Fragments',
        'groups': 'Groups',
        
        # Groups
        'groups_title': 'Grouping',
        'group_name': 'Group Name',
        'save_btn': '💾 Save to Database',
        'pieces': 'pieces',
        'no_groups': 'No groups detected',
        'saved': 'Saved!',
        
        # Reconstruction
        'reconstruction_title': 'Reconstruction',
        'select_group': 'Select Group',
        'canvas_size': 'Canvas Size',
        'calculate_btn': '🔄 Calculate',
        'matches_found': 'Matches',
        'export_btn': '📥 Save as Image',
        'edge_matches': 'Edge Matches',
        
        # Database
        'database_title': 'My Collection',
        'fragments_saved': 'Saved Fragments',
        'groups_saved': 'Saved Groups',
        'db_empty': 'No data saved yet',
        
        # Labels
        'labels_title': 'Create Labels',
        'label_source': 'Source',
        'label_session': 'Current Analysis',
        'label_database': 'Database',
        'label_custom': 'Custom Label',
        'custom_label_id': 'Custom ID (optional)',
        'custom_label_name': 'Name',
        'custom_label_desc': 'Description',
        'custom_label_project': 'Project',
        'create_pdf': '📄 Create PDF',
        'add_custom_label': '➕ Add Label',
        'custom_labels_list': 'Custom Labels',
        'clear_custom': 'Clear List',
        
        # Search
        'search_title': 'Search Fragment',
        'search_input': 'Enter Fragment ID',
        'search_btn': '🔍 Search',
        'search_found': 'Fragment found!',
        'search_not_found': 'Not found',
        
        # Demo
        'demo_title': 'Try the Demo',
        'demo_desc': 'Download a test image:',
        'demo_pottery': 'Pottery Shards',
        'demo_plate': 'Broken Plate',
        'demo_download': '📥 Download',
        'demo_hint': 'After download: Upload on the left → Analyze',
        
        # Help
        'help_title': 'Guide',
        
        # General
        'upload_first': 'Please upload photos first',
        'success_fragments': 'fragments detected!',
        'language': 'Language',
    }
}

def t(key):
    lang = st.session_state.get('language', 'de')
    return TRANSLATIONS.get(lang, TRANSLATIONS['de']).get(key, key)


# =============================================================================
# HELP CONTENT
# =============================================================================

HELP_DE = """
## Schnellstart

1. **Demo testen**: Lade ein Testbild im Start-Tab herunter
2. **Hochladen**: Ziehe das Bild in den Upload-Bereich (links)
3. **Analysieren**: Klicke auf "🔬 Analysieren"
4. **Gruppen**: Zusammengehörige Fragmente werden automatisch gruppiert
5. **Rekonstruieren**: Wähle eine Gruppe und berechne die Rekonstruktion

## Tipps für gute Ergebnisse

✅ **Heller, einfarbiger Hintergrund** (weiß, grau, beige)  
✅ **Gute Beleuchtung** ohne harte Schatten  
✅ **Fragmente nicht überlappen** lassen  
✅ **Kamera senkrecht** von oben  

## Erkennungsmodus

- **Automatisch**: Versucht den besten Modus zu finden
- **Heller Hintergrund**: Für dunkle Scherben auf hellem Untergrund
- **Dunkler Hintergrund**: Für helle Scherben auf dunklem Untergrund
- **Hoher Kontrast**: Für schwierige Lichtverhältnisse

## QR-Codes & Etiketten

Im Tab "🏷️ Etiketten" kannst du:
- PDF-Labels für analysierte Fragmente erstellen
- Eigene Etiketten für rekonstruierte Objekte erstellen
- QR-Codes führen direkt zum Fragment in der App
"""

HELP_EN = """
## Quick Start

1. **Try Demo**: Download a test image in the Start tab
2. **Upload**: Drag the image to the upload area (left)
3. **Analyze**: Click "🔬 Analyze"
4. **Groups**: Related fragments are automatically grouped
5. **Reconstruct**: Select a group and calculate reconstruction

## Tips for Good Results

✅ **Light, solid background** (white, gray, beige)  
✅ **Good lighting** without harsh shadows  
✅ **Don't overlap fragments**  
✅ **Camera perpendicular** from above  

## Detection Mode

- **Automatic**: Tries to find the best mode
- **Light Background**: For dark shards on light surface
- **Dark Background**: For light shards on dark surface
- **High Contrast**: For difficult lighting conditions

## QR Codes & Labels

In the "🏷️ Labels" tab you can:
- Create PDF labels for analyzed fragments
- Create custom labels for reconstructed objects
- QR codes link directly to the fragment in the app
"""


# =============================================================================
# DEMO IMAGES (IMPROVED - Fixed positions, better separation)
# =============================================================================

def generate_demo_pottery(num=6):
    """Generate pottery shards demo - well separated"""
    np.random.seed(42)  # Reproducible
    img = np.ones((800, 800, 3), dtype=np.uint8) * 235  # Light gray background
    
    # Add subtle texture to background
    noise = np.random.randint(0, 8, (800, 800, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    colors = [
        (65, 85, 140),   # Terracotta (BGR)
        (55, 75, 125),   
        (75, 100, 155),  
        (45, 65, 110),   
        (85, 110, 165),
        (50, 70, 120),
    ]
    
    # Fixed positions - well separated
    positions = [
        (150, 180), (400, 150), (650, 200),
        (180, 480), (450, 520), (680, 450),
    ]
    
    for i in range(min(num, len(positions))):
        cx, cy = positions[i]
        
        # Irregular shard shape
        num_pts = np.random.randint(5, 8)
        angles = sorted(np.random.uniform(0, 2*np.pi, num_pts))
        
        pts = []
        for angle in angles:
            r = np.random.randint(40, 80)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            pts.append([x, y])
        
        pts = np.array(pts, dtype=np.int32)
        color = colors[i % len(colors)]
        
        # Fill
        cv2.fillPoly(img, [pts], color)
        # Edge
        edge_color = (max(0, color[0]-30), max(0, color[1]-30), max(0, color[2]-30))
        cv2.polylines(img, [pts], True, edge_color, 2)
        
        # Texture lines
        for _ in range(4):
            x1 = cx + np.random.randint(-25, 25)
            y1 = cy + np.random.randint(-25, 25)
            x2 = x1 + np.random.randint(-15, 15)
            y2 = y1 + np.random.randint(-15, 15)
            cv2.line(img, (x1, y1), (x2, y2), edge_color, 1)
    
    return img


def generate_demo_plate(num=5):
    """Generate broken plate demo - clear segments"""
    np.random.seed(123)
    img = np.ones((800, 800, 3), dtype=np.uint8) * 230
    
    # Add texture
    noise = np.random.randint(0, 10, (800, 800, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    center = 400
    outer_r = 160
    
    # Break angles
    angles = [0, 1.2, 2.5, 4.0, 5.3, 6.28]
    
    plate_color = (245, 245, 250)
    rim_color = (180, 165, 155)
    
    for i in range(len(angles) - 1):
        a1, a2 = angles[i], angles[i + 1]
        
        # Offset for "broken" effect
        ox = int(np.random.uniform(-50, 50))
        oy = int(np.random.uniform(-50, 50))
        rot = np.random.uniform(-0.1, 0.1)
        
        pts = []
        # Inner arc
        for a in np.linspace(a1 + rot, a2 + rot, 12):
            x = int(center + ox + 25 * np.cos(a))
            y = int(center + oy + 25 * np.sin(a))
            pts.append([x, y])
        
        # Outer arc (reverse)
        for a in np.linspace(a2 + rot, a1 + rot, 18):
            x = int(center + ox + outer_r * np.cos(a))
            y = int(center + oy + outer_r * np.sin(a))
            pts.append([x, y])
        
        pts = np.array(pts, dtype=np.int32)
        
        cv2.fillPoly(img, [pts], plate_color)
        cv2.polylines(img, [pts], True, (190, 190, 195), 2)
        
        # Rim decoration
        for a in np.linspace(a1 + rot, a2 + rot, 6):
            rx = int(center + ox + (outer_r - 12) * np.cos(a))
            ry = int(center + oy + (outer_r - 12) * np.sin(a))
            cv2.circle(img, (rx, ry), 3, rim_color, -1)
    
    return img


def get_demo_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
    return buf.getvalue()


# =============================================================================
# USER MANAGEMENT (Accounts are PERMANENT)
# =============================================================================

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users_db():
    if USERS_DB_PATH.exists():
        try:
            with open(USERS_DB_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {'users': {}, 'version': 2}

def save_users_db(db):
    db['version'] = 2
    db['last_save'] = datetime.now().isoformat()
    with open(USERS_DB_PATH, 'wb') as f:
        pickle.dump(db, f)

def register_user(username, password):
    db = load_users_db()
    if username in db['users']:
        return False, 'register_error_exists'
    if len(password) < 4:
        return False, 'register_error_short'
    
    db['users'][username] = {
        'pw_hash': hash_password(password),
        'created': datetime.now().isoformat(),
        'data': {'pieces': {}, 'clusters': {}}
    }
    save_users_db(db)
    return True, 'register_success'

def authenticate(username, password):
    db = load_users_db()
    if username not in db['users']:
        return False
    return db['users'][username]['pw_hash'] == hash_password(password)

def get_user_data(username):
    db = load_users_db()
    if username in db['users']:
        d = db['users'][username].get('data', {})
        if 'pieces' not in d: d['pieces'] = {}
        if 'clusters' not in d: d['clusters'] = {}
        return d
    return {'pieces': {}, 'clusters': {}}

def save_user_data(username, data):
    db = load_users_db()
    if username in db['users']:
        db['users'][username]['data'] = data
        save_users_db(db)


# =============================================================================
# AI INTEGRATION (Gemini + Claude)
# =============================================================================

def describe_with_gemini(image_b64, api_key):
    """Use Google Gemini to describe a fragment"""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Describe this archaeological fragment briefly in German. Include: material (ceramic, glass, metal, stone), color, approximate size, any decorations or patterns. Keep it under 50 words."},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}}
                ]
            }]
        }
        
        response = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text.strip()
    except Exception as e:
        pass
    return None


def describe_with_claude(image_b64, api_key):
    """Use Claude to describe a fragment"""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 150,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                        {"type": "text", "text": "Describe this archaeological fragment briefly in German. Include: material, color, size estimate. Under 50 words."}
                    ]
                }]
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()['content'][0]['text'].strip()
    except:
        pass
    return None


# =============================================================================
# ID & QR CODE
# =============================================================================

def gen_id():
    return f"SM-{uuid.uuid4().hex[:8].upper()}"

def gen_qr(data, size=10):
    qr = qrcode.QRCode(version=1, box_size=size, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


# =============================================================================
# IMPROVED FRAGMENT DETECTION
# =============================================================================

def segment_fragments(image, min_area=100, project="", mode="auto"):
    """
    Improved fragment segmentation with multiple detection modes
    """
    h, w = image.shape[:2]
    
    # Preprocessing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert to different color spaces
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    
    masks = []
    
    if mode == "auto" or mode == "light_bg":
        # Method 1: Adaptive threshold (good for dark objects on light bg)
        adapt1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 21, 5)
        masks.append(adapt1)
        
        # Method 2: Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(otsu)
        
        # Method 3: LAB channel (L channel for luminance)
        l_channel = lab[:, :, 0]
        _, lab_thresh = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(lab_thresh)
    
    if mode == "auto" or mode == "dark_bg":
        # For light objects on dark background
        _, inv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(inv_otsu)
        
        adapt2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 21, 5)
        masks.append(adapt2)
    
    if mode == "auto" or mode == "high_contrast":
        # Edge-based detection
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Fill holes
        contours_edge, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(gray)
        cv2.drawContours(edge_mask, contours_edge, -1, 255, -1)
        masks.append(edge_mask)
        
        # Color-based: find non-background pixels
        # Assume background is the most common color (corners)
        corners = [image[0:50, 0:50], image[0:50, w-50:w], 
                   image[h-50:h, 0:50], image[h-50:h, w-50:w]]
        bg_colors = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_mean = np.mean(bg_colors, axis=0)
        bg_std = np.std(bg_colors, axis=0) + 10
        
        # Pixels far from background
        diff = np.abs(image.astype(float) - bg_mean)
        color_mask = np.any(diff > bg_std * 2.5, axis=2).astype(np.uint8) * 255
        masks.append(color_mask)
    
    # Combine masks
    combined = np.zeros_like(gray)
    for m in masks:
        combined = cv2.bitwise_or(combined, m)
    
    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Remove border artifacts
    border = 15
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = []
    max_area = h * w * 0.7  # Max 70% of image
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / (bh + 1e-6)
        if aspect < 0.1 or aspect > 10:
            continue
        
        # Filter by solidity (filled-ness)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.3:  # Too hollow
                continue
        
        # Extract ROI
        margin = 15
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        
        roi = image[y1:y2, x1:x2].copy()
        
        # Create mask for this piece
        piece_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(piece_mask, [c], -1, 255, -1)
        mask_roi = piece_mask[y1:y2, x1:x2].copy()
        
        # Adjust contour to ROI coordinates
        contour_rel = c.copy()
        contour_rel[:, :, 0] -= x1
        contour_rel[:, :, 1] -= y1
        
        piece = {
            'id': gen_id(),
            'contour': contour_rel,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': area,
            'excavation': project,
            'created': datetime.now().isoformat()
        }
        
        # Auto-classify
        piece['name'], piece['material'], piece['color_name'] = auto_classify(roi, mask_roi)
        
        pieces.append(piece)
    
    return pieces


def auto_classify(thumbnail, mask):
    """Auto-classify material and color"""
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
    # Color name
    if s < 25:
        if v > 180:
            color = "Weiß"
        elif v > 100:
            color = "Grau"
        else:
            color = "Schwarz"
    elif h < 10 or h > 170:
        color = "Rot"
    elif h < 25:
        color = "Orange"
    elif h < 35:
        color = "Braun"
    elif h < 80:
        color = "Grün"
    elif h < 130:
        color = "Blau"
    else:
        color = "Violett"
    
    # Material guess
    if s < 15 and v > 170:
        material = "Glas"
        obj_type = "Glasscherbe"
    elif 8 < h < 30 and s > 40:
        material = "Keramik"
        obj_type = "Keramikscherbe"
    elif s < 30 and v < 90:
        material = "Metall"
        obj_type = "Metallfragment"
    elif v > 200 and s < 30:
        material = "Porzellan"
        obj_type = "Porzellanscherbe"
    else:
        material = "Keramik"
        obj_type = "Fragment"
    
    name = f"{material}_{color}"
    return name, material, color


def get_features(piece):
    """Extract features for clustering"""
    try:
        lab = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2Lab)
        hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
        
        lab_mean, lab_std = cv2.meanStdDev(lab, mask=piece['mask'])
        hsv_mean, hsv_std = cv2.meanStdDev(hsv, mask=piece['mask'])
        
        return {
            'color': np.concatenate([lab_mean.flatten(), lab_std.flatten(), 
                                    hsv_mean.flatten(), hsv_std.flatten()])
        }
    except:
        return None


# =============================================================================
# CLUSTERING
# =============================================================================

def calc_similarity(p1, p2):
    """Calculate similarity between two pieces"""
    score = 0
    
    # Color similarity
    if 'features' in p1 and 'features' in p2:
        c1, c2 = p1['features']['color'], p2['features']['color']
        dist = np.linalg.norm(c1 - c2)
        score += max(0, 100 - dist * 2.5) * 0.5
    
    # Material match
    if p1.get('material') == p2.get('material'):
        score += 25
    
    # Size similarity
    a1, a2 = p1.get('area', 0), p2.get('area', 0)
    if a1 > 0 and a2 > 0:
        ratio = min(a1, a2) / max(a1, a2)
        score += ratio * 15
    
    return score


def cluster_pieces(pieces, threshold=35):
    """Cluster pieces by similarity"""
    n = len(pieces)
    if n < 2:
        return [0] if n == 1 else []
    
    # Build similarity matrix
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = calc_similarity(pieces[i], pieces[j])
            sim[i, j] = sim[j, i] = s
    
    # Convert to distance
    dist = 100 - sim
    np.fill_diagonal(dist, 0)
    
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=100 - threshold,
            metric='precomputed',
            linkage='average'
        )
        return clustering.fit_predict(dist)
    except:
        return [0] * n


def cluster_color(cid):
    """Get color for cluster visualization"""
    if cid < 0:
        return "#B0B0B0"
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
    return colors[cid % len(colors)]


# =============================================================================
# RECONSTRUCTION
# =============================================================================

def reconstruct_group(pieces, canvas_size=700):
    """Reconstruct a group of fragments"""
    if not pieces:
        return None, [], []
    
    n = len(pieces)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 245
    
    # Calculate matches
    matches = []
    for i in range(n):
        for j in range(i + 1, n):
            score = calc_similarity(pieces[i], pieces[j])
            if score > 20:
                matches.append({'piece_i': i, 'piece_j': j, 'score': score})
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Initialize placements
    center = canvas_size // 2
    placements = [{'x': center, 'y': center, 'rot': 0, 'scale': 1.0, 'placed': i == 0} for i in range(n)]
    
    # Place based on matches
    for m in matches:
        i, j = m['piece_i'], m['piece_j']
        if placements[i]['placed'] and not placements[j]['placed']:
            angle = np.random.uniform(0, 2 * np.pi)
            offset = 60 + m['score'] * 0.35
            placements[j]['x'] = placements[i]['x'] + offset * np.cos(angle)
            placements[j]['y'] = placements[i]['y'] + offset * np.sin(angle)
            placements[j]['rot'] = np.random.uniform(-0.15, 0.15)
            placements[j]['placed'] = True
        elif placements[j]['placed'] and not placements[i]['placed']:
            angle = np.random.uniform(0, 2 * np.pi)
            offset = 60 + m['score'] * 0.35
            placements[i]['x'] = placements[j]['x'] + offset * np.cos(angle)
            placements[i]['y'] = placements[j]['y'] + offset * np.sin(angle)
            placements[i]['rot'] = np.random.uniform(-0.15, 0.15)
            placements[i]['placed'] = True
    
    # Place remaining in circle
    unplaced = [i for i in range(n) if not placements[i]['placed']]
    if unplaced:
        step = 2 * np.pi / len(unplaced)
        for idx, pi in enumerate(unplaced):
            angle = idx * step
            placements[pi]['x'] = center + canvas_size // 3 * np.cos(angle)
            placements[pi]['y'] = center + canvas_size // 3 * np.sin(angle)
            placements[pi]['placed'] = True
    
    # Draw pieces
    for idx, p in enumerate(pieces):
        if 'thumbnail' not in p:
            continue
        try:
            thumb = p['thumbnail'].copy()
            mask = p.get('mask', np.ones(thumb.shape[:2], dtype=np.uint8) * 255)
            
            h, w = thumb.shape[:2]
            scale = min(80 / max(h, w), 1.0) * placements[idx].get('scale', 1.0)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            
            thumb_s = cv2.resize(thumb, (nw, nh))
            mask_s = cv2.resize(mask, (nw, nh))
            
            rot_deg = np.degrees(placements[idx].get('rot', 0))
            M = cv2.getRotationMatrix2D((nw // 2, nh // 2), rot_deg, 1.0)
            cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
            nw_r, nh_r = int(nh * sin_v + nw * cos_v), int(nh * cos_v + nw * sin_v)
            M[0, 2] += (nw_r - nw) / 2
            M[1, 2] += (nh_r - nh) / 2
            
            thumb_r = cv2.warpAffine(thumb_s, M, (nw_r, nh_r), borderValue=(245, 245, 245))
            mask_r = cv2.warpAffine(mask_s, M, (nw_r, nh_r))
            
            px, py = int(placements[idx]['x']) - nw_r // 2, int(placements[idx]['y']) - nh_r // 2
            x1, y1 = max(0, px), max(0, py)
            x2, y2 = min(canvas_size, px + nw_r), min(canvas_size, py + nh_r)
            sx1, sy1 = x1 - px, y1 - py
            
            if x2 > x1 and y2 > y1:
                roi = canvas[y1:y2, x1:x2]
                tr = thumb_r[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
                mr = mask_r[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
                if roi.shape == tr.shape:
                    m3 = cv2.cvtColor(mr, cv2.COLOR_GRAY2BGR) / 255.0
                    canvas[y1:y2, x1:x2] = (tr * m3 + roi * (1 - m3)).astype(np.uint8)
        except:
            continue
    
    return canvas, placements, matches


# =============================================================================
# PDF LABELS
# =============================================================================

def create_labels_pdf(pieces, username):
    """Create PDF with QR code labels"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    cols, rows = 3, 4
    cell_w, cell_h = width / cols, height / rows
    margin = 10
    qr_size = 55
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, height - 22, f"ShardMind Labels - {username}")
    c.setFont("Helvetica", 8)
    c.drawString(margin, height - 35, f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(pieces)} items")
    
    offset_y = 45
    
    for idx, piece in enumerate(pieces):
        col = idx % cols
        row = (idx // cols) % rows
        
        if idx > 0 and idx % (cols * rows) == 0:
            c.showPage()
            offset_y = 0
        
        x = col * cell_w + margin
        y = height - (row + 1) * cell_h - offset_y + margin
        
        # QR Code
        piece_id = piece.get('id', gen_id())
        url = f"{BASE_URL}/?piece={piece_id}"
        qr = gen_qr(url, 5)
        qr_buf = io.BytesIO()
        qr.save(qr_buf, format='PNG')
        qr_buf.seek(0)
        try:
            c.drawImage(ImageReader(qr_buf), x, y, width=qr_size, height=qr_size)
        except:
            pass
        
        # Thumbnail
        if 'thumbnail' in piece:
            try:
                thumb_rgb = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(thumb_rgb)
                pil.thumbnail((35, 35))
                tb = io.BytesIO()
                pil.save(tb, format='PNG')
                tb.seek(0)
                c.drawImage(ImageReader(tb), x + qr_size + 5, y + 18, width=32, height=32)
            except:
                pass
        
        # Text
        tx = x + qr_size + 5
        ty = y + qr_size - 3
        
        c.setFont("Helvetica-Bold", 7)
        c.drawString(tx, ty, piece_id[:15])
        
        c.setFont("Helvetica", 6)
        c.drawString(tx, ty - 8, piece.get('name', '')[:18])
        c.drawString(tx, ty - 15, piece.get('excavation', '')[:18])
        
        # Border
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.setDash(2, 2)
        c.rect(x - 2, y - 3, cell_w - 6, cell_h - 8)
        c.setDash()
    
    c.save()
    buffer.seek(0)
    return buffer


# =============================================================================
# STREAMLIT APP
# =============================================================================

def login_page():
    """Login/Register page"""
    st.markdown(f"# 🏺 {t('app_title')}")
    st.caption(t('app_tagline'))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        lang = st.selectbox("🌍", ['de', 'en'], format_func=lambda x: '🇩🇪 Deutsch' if x == 'de' else '🇬🇧 English',
                           index=0 if st.session_state.get('language', 'de') == 'de' else 1, key='lang_login')
        if lang != st.session_state.get('language', 'de'):
            st.session_state.language = lang
            st.rerun()
    
    tab1, tab2 = st.tabs([t('login'), t('register')])
    
    with tab1:
        with st.form("login_form"):
            user = st.text_input(t('username'))
            pw = st.text_input(t('password'), type='password')
            if st.form_submit_button(t('login_btn'), use_container_width=True):
                if authenticate(user, pw):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.rerun()
                else:
                    st.error(t('login_error'))
    
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input(t('username'), key='reg_user')
            new_pw = st.text_input(t('password'), type='password', key='reg_pw')
            confirm_pw = st.text_input(t('password_confirm'), type='password')
            
            st.info(f"ℹ️ {t('account_info')}")
            
            if st.form_submit_button(t('register_btn'), use_container_width=True):
                if new_pw != confirm_pw:
                    st.error(t('register_error_password'))
                else:
                    ok, msg = register_user(new_user, new_pw)
                    if ok:
                        st.success(t(msg))
                    else:
                        st.error(t(msg))


def main():
    st.set_page_config(
        page_title=f"ShardMind v{APP_VERSION}",
        page_icon="🏺",
        layout="wide"
    )
    
    # Initialize session state
    defaults = {
        'language': 'de',
        'logged_in': False,
        'username': '',
        'pieces': [],
        'cluster_names': {},
        'custom_labels': [],
        'ai_provider': 'none',
        'api_key': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Login required
    if not st.session_state.logged_in:
        login_page()
        return
    
    username = st.session_state.username
    user_data = get_user_data(username)
    
    # Check URL params for deep links
    query_params = st.query_params
    search_id = query_params.get('piece', None)
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown(f"## 🏺 {t('app_title')}")
        st.caption(f"v{APP_VERSION} | {t('app_subtitle')}")
        
        # User info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"👤 {username}")
        with col2:
            if st.button(t('logout_btn'), help="Logout"):
                st.session_state.logged_in = False
                st.session_state.pieces = []
                st.rerun()
        
        # Language
        lang = st.selectbox(
            t('language'),
            ['de', 'en'],
            format_func=lambda x: '🇩🇪 Deutsch' if x == 'de' else '🇬🇧 English',
            index=0 if st.session_state.language == 'de' else 1,
            key='lang_main'
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
        
        st.divider()
        
        # AI Settings
        with st.expander(t('ai_settings'), expanded=False):
            provider = st.selectbox(
                t('ai_provider'),
                ['none', 'gemini', 'claude'],
                format_func=lambda x: {'none': t('ai_none'), 'gemini': 'Google Gemini', 'claude': 'Anthropic Claude'}[x],
                key='ai_prov'
            )
            st.session_state.ai_provider = provider
            
            if provider != 'none':
                api_key = st.text_input(t('api_key'), type='password', key='api_k')
                st.session_state.api_key = api_key
        
        st.divider()
        
        # Upload
        files = st.file_uploader(
            t('upload_photos'),
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help=t('upload_hint')
        )
        
        # Detection settings
        min_area = st.slider(t('min_size'), 50, 1000, 150, help=t('min_size_help'))
        threshold = st.slider(t('cluster_sens'), 10, 80, 40, help=t('cluster_help'))
        
        mode = st.selectbox(
            t('detection_mode'),
            ['auto', 'light_bg', 'dark_bg', 'high_contrast'],
            format_func=lambda x: {
                'auto': t('mode_auto'),
                'light_bg': t('mode_light_bg'),
                'dark_bg': t('mode_dark_bg'),
                'high_contrast': t('mode_high_contrast')
            }[x]
        )
        
        project = st.text_input(t('project'), value=f"Project_{datetime.now().strftime('%Y')}")
        
        st.divider()
        
        # Analyze button
        if st.button(t('analyze_btn'), type='primary', use_container_width=True):
            if files:
                with st.spinner(t('analyzing')):
                    all_pieces = []
                    progress = st.progress(0)
                    
                    for i, f in enumerate(files):
                        img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        pieces = segment_fragments(img, min_area, project, mode)
                        all_pieces.extend(pieces)
                        progress.progress((i + 1) / len(files))
                    
                    # Extract features
                    valid = []
                    for p in all_pieces:
                        feat = get_features(p)
                        if feat:
                            p['features'] = feat
                            valid.append(p)
                    
                    st.session_state.pieces = valid
                    st.session_state.cluster_names = {}
                    st.success(f"✓ {len(valid)} {t('success_fragments')}")
                    st.rerun()
            else:
                st.warning(t('upload_first'))
        
        if st.button(t('clear_btn'), use_container_width=True):
            st.session_state.pieces = []
            st.session_state.cluster_names = {}
            if 'recon_image' in st.session_state:
                del st.session_state.recon_image
            st.rerun()
        
        st.divider()
        
        # Stats
        col1, col2 = st.columns(2)
        col1.metric(t('fragments'), len(user_data.get('pieces', {})))
        col2.metric(t('groups'), len(user_data.get('clusters', {})))
    
    # === PROCESS PIECES ===
    active = [p for p in st.session_state.pieces if not p.get('deleted')]
    
    if len(active) > 1:
        labels = cluster_pieces(active, threshold)
        for i, p in enumerate(active):
            p['cluster'] = labels[i]
    elif active:
        for p in active:
            p['cluster'] = 0
    
    cluster_ids = set(p.get('cluster', -1) for p in active) if active else set()
    n_clusters = len([c for c in cluster_ids if c >= 0])
    
    # === METRICS ===
    if active:
        col1, col2, col3 = st.columns(3)
        col1.metric(f"🏺 {t('fragments')}", len(active))
        col2.metric(f"📦 {t('groups')}", n_clusters)
        col3.metric("🗺️", project[:20])
    
    # === TABS ===
    tabs = st.tabs([
        t('tab_start'),
        t('tab_gallery'),
        t('tab_groups'),
        t('tab_reconstruction'),
        t('tab_database'),
        t('tab_labels'),
        t('tab_help')
    ])
    
    # --- TAB 0: START ---
    with tabs[0]:
        st.header(f"🏺 {t('app_title')}")
        st.write(t('app_tagline'))
        
        # Deep link search result
        if search_id:
            st.info(f"🔍 **{search_id}**")
            found = None
            for p in active:
                if p['id'] == search_id:
                    found = p
                    break
            if not found and search_id in user_data.get('pieces', {}):
                found = user_data['pieces'][search_id]
            
            if found:
                st.success(t('search_found'))
                col1, col2 = st.columns([1, 2])
                with col1:
                    if 'thumbnail' in found:
                        st.image(cv2.cvtColor(found['thumbnail'], cv2.COLOR_BGR2RGB))
                with col2:
                    st.write(f"**ID:** `{found['id']}`")
                    st.write(f"**Name:** {found.get('name', '-')}")
                    st.write(f"**Material:** {found.get('material', '-')}")
                    st.write(f"**Projekt:** {found.get('excavation', '-')}")
            else:
                st.warning(f"{t('search_not_found')}: {search_id}")
            st.divider()
        
        # Manual search
        st.subheader(t('search_title'))
        col1, col2 = st.columns([3, 1])
        with col1:
            search_input = st.text_input("ID", placeholder="SM-XXXXXXXX", label_visibility="collapsed")
        with col2:
            if st.button(t('search_btn'), use_container_width=True):
                if search_input:
                    st.query_params['piece'] = search_input.strip().upper()
                    st.rerun()
        
        st.divider()
        
        # Demo images
        st.subheader(t('demo_title'))
        st.write(t('demo_desc'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🏺 {t('demo_pottery')}**")
            demo1 = generate_demo_pottery(6)
            st.image(cv2.cvtColor(demo1, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.download_button(t('demo_download'), get_demo_bytes(demo1), "demo_pottery.png", "image/png", use_container_width=True, key="dl1")
        
        with col2:
            st.markdown(f"**🍽️ {t('demo_plate')}**")
            demo2 = generate_demo_plate(5)
            st.image(cv2.cvtColor(demo2, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.download_button(t('demo_download'), get_demo_bytes(demo2), "demo_plate.png", "image/png", use_container_width=True, key="dl2")
        
        st.info(f"💡 {t('demo_hint')}")
    
    # --- TAB 1: GALLERY ---
    with tabs[1]:
        st.header(t('detected_fragments'))
        
        if active:
            # Use st.image instead of HTML for Android compatibility
            cols = st.columns(5)
            for i, p in enumerate(active):
                with cols[i % 5]:
                    # Convert to RGB for display
                    thumb_rgb = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB)
                    
                    # Add colored border using numpy
                    color_hex = cluster_color(p.get('cluster', -1))
                    # Convert hex to BGR
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    
                    # Add border
                    bordered = cv2.copyMakeBorder(thumb_rgb, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(r, g, b))
                    
                    st.image(bordered, use_container_width=True)
                    st.caption(f"**{p['id'][:11]}**\n{p.get('name', '')[:15]}")
        else:
            st.info(t('upload_first'))
    
    # --- TAB 2: GROUPS ---
    with tabs[2]:
        st.header(t('groups_title'))
        
        if active:
            sorted_clusters = sorted([c for c in cluster_ids if c >= 0])
            
            if not sorted_clusters:
                st.warning(t('no_groups'))
            else:
                for cid in sorted_clusters:
                    cp = [p for p in active if p.get('cluster') == cid]
                    mats = [p.get('material', 'Unknown') for p in cp]
                    common_mat = max(set(mats), key=mats.count) if mats else 'Unknown'
                    default_name = st.session_state.cluster_names.get(cid, f"{common_mat}_Gruppe_{cid + 1}")
                    
                    with st.expander(f"📦 {default_name} ({len(cp)} {t('pieces')})", expanded=True):
                        name = st.text_input(t('group_name'), value=default_name, key=f"gn_{cid}")
                        st.session_state.cluster_names[cid] = name
                        
                        # Show thumbnails using st.image (Android compatible)
                        pcols = st.columns(min(6, len(cp)))
                        for i, p in enumerate(cp[:6]):
                            with pcols[i]:
                                st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.caption(p['id'][:8])
                        
                        if st.button(t('save_btn'), key=f"save_{cid}"):
                            key = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            user_data['clusters'][key] = {
                                'name': name,
                                'created': datetime.now().isoformat(),
                                'piece_ids': [p['id'] for p in cp]
                            }
                            for p in cp:
                                p_copy = {k: v for k, v in p.items() if k != 'features'}
                                user_data['pieces'][p['id']] = p_copy
                            save_user_data(username, user_data)
                            st.success(t('saved'))
                            st.rerun()
        else:
            st.info(t('upload_first'))
    
    # --- TAB 3: RECONSTRUCTION ---
    with tabs[3]:
        st.header(t('reconstruction_title'))
        
        if active and n_clusters > 0:
            opts = {
                f"{st.session_state.cluster_names.get(c, f'Gruppe_{c+1}')} ({sum(1 for p in active if p.get('cluster') == c)} {t('pieces')})": c
                for c in cluster_ids if c >= 0
            }
            
            sel_name = st.selectbox(t('select_group'), list(opts.keys()))
            sel_id = opts[sel_name]
            
            rp = [p for p in active if p.get('cluster') == sel_id]
            
            # Preview
            st.write(f"**{len(rp)} {t('pieces')}**")
            pcols = st.columns(min(8, len(rp)))
            for i, p in enumerate(rp[:8]):
                with pcols[i]:
                    st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            canvas_size = st.slider(t('canvas_size'), 400, 1000, 700, 50)
            
            if st.button(t('calculate_btn'), type='primary', use_container_width=True):
                with st.spinner(t('analyzing')):
                    img, plc, matches = reconstruct_group(rp, canvas_size)
                    if img is not None:
                        st.session_state.recon_image = img
                        st.session_state.recon_matches = matches
                        st.success(f"✓ {len(matches)} {t('matches_found')}")
            
            if 'recon_image' in st.session_state and st.session_state.recon_image is not None:
                col1, col2 = st.columns([2, 1])
                with col1:
                    rgb = cv2.cvtColor(st.session_state.recon_image, cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                    
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format='PNG')
                    st.download_button(t('export_btn'), buf.getvalue(), f"reconstruction.png", "image/png", use_container_width=True)
                
                with col2:
                    st.markdown(f"**{t('edge_matches')}**")
                    for i, m in enumerate(st.session_state.get('recon_matches', [])[:5]):
                        st.write(f"• {m['piece_i'] + 1} ↔ {m['piece_j'] + 1}: {m['score']:.0f}%")
        else:
            st.info(t('upload_first'))
    
    # --- TAB 4: DATABASE ---
    with tabs[4]:
        st.header(t('database_title'))
        
        db_pieces = user_data.get('pieces', {})
        db_clusters = user_data.get('clusters', {})
        
        if db_pieces:
            st.write(f"**{len(db_pieces)} {t('fragments_saved')}**")
            
            dcols = st.columns(6)
            for i, (pid, p) in enumerate(list(db_pieces.items())[:18]):
                with dcols[i % 6]:
                    if 'thumbnail' in p:
                        st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(pid[:11])
            
            if db_clusters:
                st.divider()
                st.write(f"**{len(db_clusters)} {t('groups_saved')}**")
                for k, v in db_clusters.items():
                    st.write(f"• {v.get('name', k)} ({len(v.get('piece_ids', []))} {t('pieces')})")
        else:
            st.info(t('db_empty'))
    
    # --- TAB 5: LABELS ---
    with tabs[5]:
        st.header(t('labels_title'))
        
        source = st.radio(
            t('label_source'),
            [t('label_session'), t('label_database'), t('label_custom')],
            horizontal=True
        )
        
        label_pieces = []
        
        if source == t('label_session'):
            label_pieces = active
        elif source == t('label_database'):
            label_pieces = list(user_data.get('pieces', {}).values())
        else:
            # Custom labels
            st.subheader(t('label_custom'))
            
            with st.form("custom_label_form"):
                col1, col2 = st.columns(2)
                with col1:
                    custom_id = st.text_input(t('custom_label_id'), placeholder="SM-CUSTOM01")
                    custom_name = st.text_input(t('custom_label_name'), placeholder="Rekonstruierter Teller")
                with col2:
                    custom_desc = st.text_input(t('custom_label_desc'), placeholder="5 Teile zusammengesetzt")
                    custom_project = st.text_input(t('custom_label_project'), value=project)
                
                if st.form_submit_button(t('add_custom_label'), use_container_width=True):
                    new_label = {
                        'id': custom_id if custom_id else gen_id(),
                        'name': custom_name or "Custom",
                        'description': custom_desc,
                        'excavation': custom_project,
                        'created': datetime.now().isoformat()
                    }
                    if 'custom_labels' not in st.session_state:
                        st.session_state.custom_labels = []
                    st.session_state.custom_labels.append(new_label)
                    st.success(t('saved'))
                    st.rerun()
            
            # Show custom labels
            if st.session_state.get('custom_labels'):
                st.subheader(t('custom_labels_list'))
                for i, lbl in enumerate(st.session_state.custom_labels):
                    st.write(f"• **{lbl['id']}** - {lbl['name']}")
                
                if st.button(t('clear_custom')):
                    st.session_state.custom_labels = []
                    st.rerun()
                
                label_pieces = st.session_state.custom_labels
        
        if label_pieces:
            st.write(f"**{len(label_pieces)} Labels**")
            
            # Preview
            if source != t('label_custom'):
                pcols = st.columns(min(6, len(label_pieces)))
                for i, p in enumerate(label_pieces[:6]):
                    with pcols[i]:
                        if 'thumbnail' in p:
                            st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), width=60)
                        st.caption(p.get('id', '')[:8])
            
            if st.button(t('create_pdf'), type='primary', use_container_width=True):
                with st.spinner("..."):
                    pdf = create_labels_pdf(label_pieces, username)
                    st.download_button(
                        "📥 PDF",
                        pdf.getvalue(),
                        f"shardmind_labels_{datetime.now().strftime('%Y%m%d')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
        else:
            st.info(t('upload_first'))
    
    # --- TAB 6: HELP ---
    with tabs[6]:
        st.header(t('help_title'))
        st.markdown(HELP_DE if st.session_state.language == 'de' else HELP_EN)


if __name__ == "__main__":
    main()
