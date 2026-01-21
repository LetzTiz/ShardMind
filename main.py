"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗  ██╗ █████╗ ██████╗ ██████╗ ███╗   ███╗██╗███╗   ██╗██████╗    ║
║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗   ║
║   ███████╗███████║███████║██████╔╝██║  ██║██╔████╔██║██║██╔██╗ ██║██║  ██║   ║
║   ╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║   ║
║   ███████║██║  ██║██║  ██║██║  ██║██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██████╔╝   ║
║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝    ║
║                                                                               ║
║   Archäologische Fragmentanalyse & Rekonstruktion                            ║
║   Archaeological Fragment Analysis & Reconstruction                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG:
==========

Version 1.7 (2025-01-21)
------------------------
NEU:
• 🎯 OPTIMIERTE SEGMENTIERUNG für echte Tellerscherben:
  - Schatten-Erkennung (findet weiße Teile auf weißem Hintergrund)
  - Farb-Erkennung für Dekorbänder (Blau, Grün)
  - Neuer Modus "Porzellan/Keramik"
• 📊 VERBESSERTE KLASSIFIZIERUNG:
  - Erkennt Randstücke (mit Dekor + Krümmung) vs. Mittelstücke
  - Analysiert Krümmungsradius für Positionsbestimmung
• 🧩 TELLER-REKONSTRUKTION:
  - Ordnet Randstücke im Kreis an
  - Mittelstücke werden ins Zentrum platziert
  - Basiert auf Original-Position im Foto

Version 1.5 (2025-01-21)
------------------------
NEU:
• 🧩 KOMPLETT NEUE REKONSTRUKTION mit echter Kantenpassung
  - Analysiert Konturen und findet passende Kanten
  - Berechnet optimale Rotation und Position
  - Iteratives Zusammensetzen der Fragmente
• 📄 PDF-Layout komplett überarbeitet
  - Korrektes 3x4 Raster auf A4
  - Bessere Abstände und Lesbarkeit
  - QR-Codes und Thumbnails richtig positioniert
• 📋 Changelog zurück im Code

Version 1.4 (2025-01-21)
------------------------
• 🤖 Gemini API Integration (zusätzlich zu Claude)
• 💾 Accounts werden DAUERHAFT gespeichert (keine Löschung)
• 🌍 Verbesserte Übersetzungen (DE/EN)
• 🏷️ Eigene Etiketten für rekonstruierte Objekte erstellen
• 🔍 Verbesserte Fragment-Erkennung (4 Modi)
• 📱 Android Bild-Bug behoben (kein base64 HTML mehr)
• 🔗 QR-Scanner ersetzt durch ID-Eingabe + Deep Links

Version 1.3 (2025-01-21)
------------------------
• 🏠 Start-Tab mit Demo-Bildern und QR-Suche
• 🏷️ PDF-Etiketten mit QR-Codes
• 📊 Verbessertes Clustering (4-Faktor-Scoring)
• 🔗 Deep-Link QR-Codes (URLs statt App-Schema)

Version 1.2 (2025-01-16)
------------------------
• 🔐 Login-System mit privater Datenbank
• 🧩 Kanten-basiertes Clustering
• 📖 Hilfe-Tab
• 🌍 Zweisprachig (Deutsch/English)

Version 1.1 (2025-01-15)
------------------------
• 🧩 Erste Rekonstruktions-Funktion
• 📦 Gruppierung nach Farbe/Material

Version 1.0 (2025-01-14)
------------------------
• 🏺 Erste Version
• 🔬 Grundlegende Segmentierung
• 🎨 Farbanalyse

"""

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.ndimage import rotate as ndimage_rotate
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
from reportlab.lib.units import mm
import uuid
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_VERSION = "1.7"
USERS_DB_PATH = Path("shardmind_users.pkl")
BASE_URL = "https://shardmind.streamlit.app"

# =============================================================================
# TRANSLATIONS
# =============================================================================

TRANSLATIONS = {
    'de': {
        'app_title': 'ShardMind',
        'app_subtitle': 'Archäologische Fragmentanalyse',
        'app_tagline': 'Analysiere und rekonstruiere zerbrochene Objekte',
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
        'settings': '⚙️ Einstellungen',
        'ai_settings': '🤖 KI-Einstellungen',
        'ai_provider': 'KI-Anbieter',
        'api_key': 'API-Schlüssel',
        'ai_none': 'Keine KI',
        'upload_photos': '📤 Fotos hochladen',
        'upload_hint': 'JPG, PNG - Scherben auf hellem Hintergrund',
        'min_size': 'Min. Fragmentgröße',
        'cluster_sens': 'Gruppierungsstärke',
        'detection_mode': 'Erkennungsmodus',
        'mode_auto': 'Automatisch',
        'mode_light_bg': 'Heller Hintergrund',
        'mode_dark_bg': 'Dunkler Hintergrund',
        'mode_high_contrast': 'Hoher Kontrast',
        'mode_porcelain': 'Porzellan/Keramik',
        'edge_pieces': 'Randstücke',
        'center_pieces': 'Mittelstücke',
        'project': 'Projekt/Grabung',
        'analyze_btn': '🔬 Analysieren',
        'clear_btn': '🗑️ Leeren',
        'analyzing': 'Analysiere...',
        'tab_start': '🏠 Start',
        'tab_gallery': '🏺 Galerie',
        'tab_groups': '📦 Gruppen',
        'tab_reconstruction': '🧩 Rekonstruktion',
        'tab_database': '💾 Datenbank',
        'tab_labels': '🏷️ Etiketten',
        'tab_help': '❓ Hilfe',
        'detected_fragments': 'Erkannte Fragmente',
        'fragments': 'Fragmente',
        'groups': 'Gruppen',
        'groups_title': 'Gruppierung',
        'group_name': 'Gruppenname',
        'save_btn': '💾 Speichern',
        'pieces': 'Teile',
        'no_groups': 'Keine Gruppen',
        'saved': 'Gespeichert!',
        'reconstruction_title': 'Rekonstruktion',
        'select_group': 'Gruppe wählen',
        'canvas_size': 'Größe',
        'calculate_btn': '🧩 Zusammensetzen',
        'matches_found': 'Verbindungen',
        'export_btn': '📥 Bild speichern',
        'edge_matches': 'Kantenpassungen',
        'database_title': 'Meine Sammlung',
        'fragments_saved': 'Fragmente',
        'groups_saved': 'Gruppen',
        'db_empty': 'Noch leer',
        'labels_title': 'Etiketten erstellen',
        'label_source': 'Quelle',
        'label_session': 'Aktuelle Analyse',
        'label_database': 'Datenbank',
        'label_custom': 'Eigenes Etikett',
        'custom_label_name': 'Bezeichnung',
        'custom_label_desc': 'Beschreibung',
        'create_pdf': '📄 PDF erstellen',
        'add_custom_label': '➕ Hinzufügen',
        'custom_labels_list': 'Eigene Etiketten',
        'clear_custom': 'Liste leeren',
        'search_title': 'Fragment suchen',
        'search_btn': '🔍 Suchen',
        'search_found': 'Gefunden!',
        'search_not_found': 'Nicht gefunden',
        'demo_title': 'Demo',
        'demo_pottery': 'Keramik',
        'demo_plate': 'Teller',
        'demo_download': '📥 Download',
        'demo_hint': 'Herunterladen → Links hochladen → Analysieren',
        'help_title': 'Anleitung',
        'upload_first': 'Bitte zuerst Fotos hochladen',
        'success_fragments': 'Fragmente erkannt!',
        'language': 'Sprache',
        'recon_step': 'Schritt',
        'recon_placing': 'Platziere Fragment',
        'recon_complete': 'Rekonstruktion abgeschlossen',
    },
    'en': {
        'app_title': 'ShardMind',
        'app_subtitle': 'Archaeological Fragment Analysis',
        'app_tagline': 'Analyze and reconstruct broken objects',
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
        'settings': '⚙️ Settings',
        'ai_settings': '🤖 AI Settings',
        'ai_provider': 'AI Provider',
        'api_key': 'API Key',
        'ai_none': 'No AI',
        'upload_photos': '📤 Upload Photos',
        'upload_hint': 'JPG, PNG - Fragments on light background',
        'min_size': 'Min. Fragment Size',
        'cluster_sens': 'Grouping Strength',
        'detection_mode': 'Detection Mode',
        'mode_auto': 'Automatic',
        'mode_light_bg': 'Light Background',
        'mode_dark_bg': 'Dark Background',
        'mode_high_contrast': 'High Contrast',
        'mode_porcelain': 'Porcelain/Ceramic',
        'edge_pieces': 'Edge pieces',
        'center_pieces': 'Center pieces',
        'project': 'Project/Excavation',
        'analyze_btn': '🔬 Analyze',
        'clear_btn': '🗑️ Clear',
        'analyzing': 'Analyzing...',
        'tab_start': '🏠 Start',
        'tab_gallery': '🏺 Gallery',
        'tab_groups': '📦 Groups',
        'tab_reconstruction': '🧩 Reconstruction',
        'tab_database': '💾 Database',
        'tab_labels': '🏷️ Labels',
        'tab_help': '❓ Help',
        'detected_fragments': 'Detected Fragments',
        'fragments': 'Fragments',
        'groups': 'Groups',
        'groups_title': 'Grouping',
        'group_name': 'Group Name',
        'save_btn': '💾 Save',
        'pieces': 'pieces',
        'no_groups': 'No groups',
        'saved': 'Saved!',
        'reconstruction_title': 'Reconstruction',
        'select_group': 'Select Group',
        'canvas_size': 'Size',
        'calculate_btn': '🧩 Reconstruct',
        'matches_found': 'Connections',
        'export_btn': '📥 Save Image',
        'edge_matches': 'Edge Matches',
        'database_title': 'My Collection',
        'fragments_saved': 'Fragments',
        'groups_saved': 'Groups',
        'db_empty': 'Empty',
        'labels_title': 'Create Labels',
        'label_source': 'Source',
        'label_session': 'Current Analysis',
        'label_database': 'Database',
        'label_custom': 'Custom Label',
        'custom_label_name': 'Name',
        'custom_label_desc': 'Description',
        'create_pdf': '📄 Create PDF',
        'add_custom_label': '➕ Add',
        'custom_labels_list': 'Custom Labels',
        'clear_custom': 'Clear List',
        'search_title': 'Search Fragment',
        'search_btn': '🔍 Search',
        'search_found': 'Found!',
        'search_not_found': 'Not found',
        'demo_title': 'Demo',
        'demo_pottery': 'Pottery',
        'demo_plate': 'Plate',
        'demo_download': '📥 Download',
        'demo_hint': 'Download → Upload left → Analyze',
        'help_title': 'Guide',
        'upload_first': 'Please upload photos first',
        'success_fragments': 'fragments detected!',
        'language': 'Language',
        'recon_step': 'Step',
        'recon_placing': 'Placing fragment',
        'recon_complete': 'Reconstruction complete',
    }
}

def t(key):
    lang = st.session_state.get('language', 'de')
    return TRANSLATIONS.get(lang, TRANSLATIONS['de']).get(key, key)


# =============================================================================
# HELP
# =============================================================================

HELP_DE = """
## Schnellstart

1. **Demo testen**: Lade ein Testbild im Start-Tab herunter
2. **Hochladen**: Ziehe das Bild in den Upload-Bereich
3. **Analysieren**: Klicke auf "🔬 Analysieren"
4. **Rekonstruieren**: Wähle eine Gruppe → "🧩 Zusammensetzen"

## Tipps

- **Heller Hintergrund** (weiß, grau) funktioniert am besten
- **Nicht überlappen** lassen
- **Gute Beleuchtung** ohne Schatten
- Bei Problemen: Erkennungsmodus ändern

## Rekonstruktion

Die neue Rekonstruktion analysiert die **Kanten** der Fragmente:
- Findet passende Kantenabschnitte
- Berechnet optimale Rotation
- Setzt Teile schrittweise zusammen

## Etiketten

Im Tab "🏷️ Etiketten":
- PDF mit QR-Codes erstellen
- Eigene Etiketten für rekonstruierte Objekte
- QR-Codes verlinken direkt zur App
"""

HELP_EN = """
## Quick Start

1. **Try Demo**: Download a test image in Start tab
2. **Upload**: Drag image to upload area
3. **Analyze**: Click "🔬 Analyze"
4. **Reconstruct**: Select group → "🧩 Reconstruct"

## Tips

- **Light background** (white, gray) works best
- **Don't overlap** fragments
- **Good lighting** without shadows
- If problems: Change detection mode

## Reconstruction

The new reconstruction analyzes fragment **edges**:
- Finds matching edge segments
- Calculates optimal rotation
- Assembles pieces step by step

## Labels

In "🏷️ Labels" tab:
- Create PDF with QR codes
- Custom labels for reconstructed objects
- QR codes link directly to app
"""


# =============================================================================
# DEMO IMAGES - Puzzle pieces that actually fit together
# =============================================================================

def generate_demo_pottery(num=6):
    """Generate pottery shards that can be reconstructed"""
    np.random.seed(42)
    img = np.ones((800, 800, 3), dtype=np.uint8) * 230
    
    # Create a complete circle/pot shape first, then break it
    center_x, center_y = 400, 400
    radius = 150
    
    # Create the original shape
    original = np.zeros((800, 800), dtype=np.uint8)
    cv2.circle(original, (center_x, center_y), radius, 255, -1)
    
    # Break into pieces with irregular lines from center
    angles = np.linspace(0, 2*np.pi, num + 1)[:-1]
    angles += np.random.uniform(-0.3, 0.3, num)
    angles = np.sort(angles)
    
    colors = [
        (55, 75, 130),   # Terracotta BGR
        (60, 80, 135),
        (50, 70, 125),
        (65, 85, 140),
        (45, 65, 120),
        (70, 90, 145),
    ]
    
    pieces_data = []
    
    for i in range(num):
        a1 = angles[i]
        a2 = angles[(i + 1) % num]
        if a2 < a1:
            a2 += 2 * np.pi
        
        # Create piece mask
        piece_mask = np.zeros((800, 800), dtype=np.uint8)
        
        # Pie slice from center
        pts = [(center_x, center_y)]
        for a in np.linspace(a1, a2, 20):
            x = int(center_x + (radius + 10) * np.cos(a))
            y = int(center_y + (radius + 10) * np.sin(a))
            pts.append((x, y))
        pts.append((center_x, center_y))
        
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(piece_mask, [pts], 255)
        
        # Intersect with original shape
        piece_mask = cv2.bitwise_and(piece_mask, original)
        
        # Add some irregularity to edges (break lines)
        kernel = np.ones((3, 3), np.uint8)
        piece_mask = cv2.erode(piece_mask, kernel, iterations=2)
        
        if cv2.countNonZero(piece_mask) < 500:
            continue
        
        # Find contour
        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate offset position (spread pieces apart)
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Offset from center
        offset_angle = (a1 + a2) / 2
        offset_dist = 80 + np.random.randint(0, 40)
        ox = int(offset_dist * np.cos(offset_angle))
        oy = int(offset_dist * np.sin(offset_angle))
        
        # Shift contour
        shifted_contour = contour.copy()
        shifted_contour[:, :, 0] += ox
        shifted_contour[:, :, 1] += oy
        
        # Draw on image
        color = colors[i % len(colors)]
        cv2.fillPoly(img, [shifted_contour], color)
        
        # Add edge highlight
        edge_color = (max(0, color[0] - 25), max(0, color[1] - 25), max(0, color[2] - 25))
        cv2.polylines(img, [shifted_contour], True, edge_color, 2)
        
        # Add texture
        x, y, w, h = cv2.boundingRect(shifted_contour)
        for _ in range(8):
            tx = x + np.random.randint(10, max(11, w - 10))
            ty = y + np.random.randint(10, max(11, h - 10))
            if piece_mask[min(ty - oy, 799), min(tx - ox, 799)] > 0:
                cv2.line(img, (tx, ty), (tx + np.random.randint(-8, 8), ty + np.random.randint(-8, 8)), edge_color, 1)
        
        pieces_data.append({
            'contour': shifted_contour,
            'original_pos': (cx, cy),
            'offset': (ox, oy)
        })
    
    return img


def generate_demo_plate(num=5):
    """Generate broken plate pieces that fit together"""
    np.random.seed(123)
    img = np.ones((800, 800, 3), dtype=np.uint8) * 225
    
    center_x, center_y = 400, 400
    outer_r = 160
    inner_r = 30
    
    # Create plate shape
    original = np.zeros((800, 800), dtype=np.uint8)
    cv2.circle(original, (center_x, center_y), outer_r, 255, -1)
    cv2.circle(original, (center_x, center_y), inner_r, 0, -1)  # Hole in center
    
    # Break angles
    angles = np.linspace(0, 2*np.pi, num + 1)[:-1]
    angles += np.random.uniform(-0.2, 0.2, num)
    angles = np.sort(angles)
    
    plate_color = (250, 248, 245)  # Off-white
    rim_color = (165, 155, 140)    # Decorative rim
    
    for i in range(num):
        a1 = angles[i]
        a2 = angles[(i + 1) % num]
        if a2 < a1:
            a2 += 2 * np.pi
        
        # Create piece
        piece_mask = np.zeros((800, 800), dtype=np.uint8)
        
        pts = []
        # Inner arc
        for a in np.linspace(a1, a2, 15):
            x = int(center_x + inner_r * np.cos(a))
            y = int(center_y + inner_r * np.sin(a))
            pts.append([x, y])
        
        # Outer arc (reverse)
        for a in np.linspace(a2, a1, 25):
            x = int(center_x + outer_r * np.cos(a))
            y = int(center_y + outer_r * np.sin(a))
            pts.append([x, y])
        
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(piece_mask, [pts], 255)
        
        # Find contour
        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        
        # Offset
        mid_angle = (a1 + a2) / 2
        offset_dist = 60 + np.random.randint(0, 50)
        ox = int(offset_dist * np.cos(mid_angle))
        oy = int(offset_dist * np.sin(mid_angle))
        
        shifted = contour.copy()
        shifted[:, :, 0] += ox
        shifted[:, :, 1] += oy
        
        # Draw
        cv2.fillPoly(img, [shifted], plate_color)
        cv2.polylines(img, [shifted], True, (200, 195, 190), 2)
        
        # Rim decoration
        for a in np.linspace(a1, a2, 8):
            rx = int(center_x + ox + (outer_r - 15) * np.cos(a))
            ry = int(center_y + oy + (outer_r - 15) * np.sin(a))
            cv2.circle(img, (rx, ry), 4, rim_color, -1)
    
    return img


def get_demo_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
    return buf.getvalue()


# =============================================================================
# USER MANAGEMENT
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
        if 'pieces' not in d:
            d['pieces'] = {}
        if 'clusters' not in d:
            d['clusters'] = {}
        return d
    return {'pieces': {}, 'clusters': {}}

def save_user_data(username, data):
    db = load_users_db()
    if username in db['users']:
        db['users'][username]['data'] = data
        save_users_db(db)


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
# FRAGMENT DETECTION
# =============================================================================

def segment_fragments(image, min_area=100, project="", mode="auto"):
    """Improved fragment detection"""
    h, w = image.shape[:2]
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    
    masks = []
    
    if mode in ["auto", "light_bg", "porcelain"]:
        # Adaptive threshold
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 25, 8)
        masks.append(adapt)
        
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(otsu)
        
        # LAB L-channel
        l_ch = lab[:, :, 0]
        _, lab_th = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(lab_th)
    
    if mode in ["auto", "dark_bg"]:
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(inv)
    
    # === PORCELAIN SPECIAL: Shadow and Color Detection ===
    if mode in ["porcelain"]:
        # Analyze background from corners
        margin = 30
        corners = [
            gray[margin:2*margin, margin:2*margin],
            gray[margin:2*margin, w-2*margin:w-margin],
            gray[h-2*margin:h-margin, margin:2*margin],
            gray[h-2*margin:h-margin, w-2*margin:w-margin]
        ]
        bg_mean = np.mean([c.mean() for c in corners])
        bg_std = np.mean([c.std() for c in corners])
        
        # Shadow detection (pieces cast shadows)
        shadow_thresh = bg_mean - 20 - bg_std
        shadow_mask = (gray < shadow_thresh).astype(np.uint8) * 255
        masks.append(shadow_mask)
        
        # L-channel threshold for subtle differences
        l_channel = lab[:, :, 0]
        l_thresh = np.percentile(l_channel, 15)
        l_mask = (l_channel < l_thresh).astype(np.uint8) * 255
        masks.append(l_mask)
        
        # Color detection (decorations)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Blue decoration
        blue_mask = cv2.inRange(hsv, np.array([90, 25, 40]), np.array([130, 255, 255]))
        # Green decoration
        green_mask = cv2.inRange(hsv, np.array([35, 25, 40]), np.array([85, 255, 255]))
        # Any saturated color
        sat_mask = (hsv[:, :, 1] > 30).astype(np.uint8) * 255
        
        color_mask = cv2.bitwise_or(blue_mask, green_mask)
        color_mask = cv2.bitwise_or(color_mask, sat_mask)
        
        # Dilate color mask to include surrounding area
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        color_expanded = cv2.dilate(color_mask, kernel_dilate, iterations=2)
        masks.append(color_expanded)
    
    if mode in ["auto", "high_contrast"]:
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges = cv2.dilate(edges, kernel, iterations=3)
        
        contours_e, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(gray)
        cv2.drawContours(edge_mask, contours_e, -1, 255, -1)
        masks.append(edge_mask)
        
        # Background subtraction
        corners = [image[0:30, 0:30], image[0:30, w-30:w], 
                   image[h-30:h, 0:30], image[h-30:h, w-30:w]]
        bg_colors = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_mean = np.mean(bg_colors, axis=0)
        bg_std = np.std(bg_colors, axis=0) + 15
        
        diff = np.abs(image.astype(float) - bg_mean)
        fg_mask = np.any(diff > bg_std * 2, axis=2).astype(np.uint8) * 255
        masks.append(fg_mask)
    
    # Combine
    combined = np.zeros_like(gray)
    for m in masks:
        combined = cv2.bitwise_or(combined, m)
    
    # Cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Remove border
    border = 10
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = []
    max_area = h * w * 0.6
    
    for c in contours:
        area = cv2.contourArea(c)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / (bh + 1e-6)
        if aspect < 0.15 or aspect > 7:
            continue
        
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.35:
                continue
        
        # Extract ROI
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        
        roi = image[y1:y2, x1:x2].copy()
        
        # Mask
        piece_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(piece_mask, [c], -1, 255, -1)
        mask_roi = piece_mask[y1:y2, x1:x2].copy()
        
        # Relative contour
        contour_rel = c.copy()
        contour_rel[:, :, 0] -= x1
        contour_rel[:, :, 1] -= y1
        
        # Centroid
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + bw // 2, y + bh // 2
        
        piece = {
            'id': gen_id(),
            'contour': contour_rel,
            'contour_global': c,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': area,
            'centroid': (cx, cy),
            'bbox': (x, y, bw, bh),
            'excavation': project,
            'created': datetime.now().isoformat()
        }
        
        # Analyze decoration and curvature for reconstruction
        piece['has_decoration'] = analyze_decoration(roi, mask_roi)
        piece['curvature'] = analyze_curvature(c)
        piece['is_edge_piece'] = piece['has_decoration'] or piece['curvature'] > 0.005
        
        # Analyze pattern position for reconstruction
        # Find the centroid of colored (blue/green) areas
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch = hsv_roi[:, :, 0]
        s_ch = hsv_roi[:, :, 1]
        
        # Blue and green detection
        blue_mask_local = ((h_ch > 90) & (h_ch < 130) & (s_ch > 40) & (mask_roi > 0)).astype(np.uint8) * 255
        green_mask_local = ((h_ch > 35) & (h_ch < 90) & (s_ch > 30) & (mask_roi > 0)).astype(np.uint8) * 255
        pattern_mask = cv2.bitwise_or(blue_mask_local, green_mask_local)
        
        pattern_M = cv2.moments(pattern_mask)
        piece['has_pattern'] = pattern_M['m00'] > 500
        
        if piece['has_pattern']:
            # Pattern center relative to piece
            pattern_cx_local = pattern_M['m10'] / pattern_M['m00']
            pattern_cy_local = pattern_M['m01'] / pattern_M['m00']
            
            # Piece center in local coords
            piece_cx_local = cx - x1
            piece_cy_local = cy - y1
            
            # Direction from piece center to pattern center = OUTWARD direction
            dx = pattern_cx_local - piece_cx_local
            dy = pattern_cy_local - piece_cy_local
            
            if abs(dx) > 2 or abs(dy) > 2:
                piece['pattern_direction'] = np.arctan2(dy, dx)
                piece['pattern_center'] = (pattern_cx_local + x1, pattern_cy_local + y1)
            else:
                piece['pattern_direction'] = None
                piece['pattern_center'] = None
        else:
            piece['pattern_direction'] = None
            piece['pattern_center'] = None
        
        # Classify
        piece['name'], piece['material'], piece['color_name'] = auto_classify(roi, mask_roi)
        
        pieces.append(piece)
    
    return pieces


def analyze_decoration(roi, mask):
    """Analyze if piece has colored decoration (blue/green bands etc)"""
    if roi is None or mask is None:
        return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colored_mask = (sat > 30) & (mask > 0)
    
    total_pixels = np.sum(mask > 0)
    colored_pixels = np.sum(colored_mask)
    
    if total_pixels == 0:
        return False
    
    ratio = colored_pixels / total_pixels
    return ratio > 0.05  # More than 5% colored = has decoration


def analyze_curvature(contour):
    """Analyze the curvature of the contour (higher = more curved = edge piece)"""
    if contour is None or len(contour) < 10:
        return 0.0
    
    pts = contour.reshape(-1, 2).astype(float)
    n = len(pts)
    
    curvatures = []
    step = max(1, n // 20)
    
    for i in range(0, n, step):
        p_prev = pts[(i - step) % n]
        p_curr = pts[i]
        p_next = pts[(i + step) % n]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        
        if l1 > 0 and l2 > 0:
            curv = cross / (l1 * l2 * (l1 + l2) / 2 + 1e-9)
            curvatures.append(curv)
    
    return np.median(curvatures) if curvatures else 0.0


def auto_classify(thumbnail, mask):
    """Auto-classify material and color"""
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
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
    elif h < 40:
        color = "Braun"
    elif h < 80:
        color = "Grün"
    elif h < 130:
        color = "Blau"
    else:
        color = "Violett"
    
    if s < 20 and v > 200:
        material = "Porzellan"
    elif 10 < h < 35 and s > 30:
        material = "Keramik"
    elif s < 25 and v < 100:
        material = "Metall"
    else:
        material = "Keramik"
    
    return f"{material}_{color}", material, color


def get_features(piece):
    """Extract features for clustering"""
    try:
        lab = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2Lab)
        lab_mean, lab_std = cv2.meanStdDev(lab, mask=piece['mask'])
        
        # Edge features
        contour = piece.get('contour', piece.get('contour_global'))
        if contour is not None:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = piece['area'] / (hull_area + 1) if hull_area > 0 else 0
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * piece['area'] / (perimeter ** 2 + 1)
        else:
            solidity, circularity = 0.5, 0.5
        
        return {
            'color': np.concatenate([lab_mean.flatten(), lab_std.flatten()]),
            'solidity': solidity,
            'circularity': circularity
        }
    except:
        return None


# =============================================================================
# CLUSTERING
# =============================================================================

def calc_similarity(p1, p2):
    """Calculate similarity between two pieces"""
    score = 0
    
    # Color
    if 'features' in p1 and 'features' in p2:
        c1, c2 = p1['features']['color'], p2['features']['color']
        dist = np.linalg.norm(c1 - c2)
        score += max(0, 100 - dist * 2) * 0.4
    
    # Material
    if p1.get('material') == p2.get('material'):
        score += 30
    
    # Size
    a1, a2 = p1.get('area', 0), p2.get('area', 0)
    if a1 > 0 and a2 > 0:
        ratio = min(a1, a2) / max(a1, a2)
        score += ratio * 20
    
    # Shape similarity
    if 'features' in p1 and 'features' in p2:
        sol_diff = abs(p1['features'].get('solidity', 0) - p2['features'].get('solidity', 0))
        score += max(0, 10 - sol_diff * 20)
    
    return score


def cluster_pieces(pieces, threshold=35):
    """Cluster pieces by similarity"""
    n = len(pieces)
    if n < 2:
        return [0] if n == 1 else []
    
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = calc_similarity(pieces[i], pieces[j])
            sim[i, j] = sim[j, i] = s
    
    dist = 100 - sim
    np.fill_diagonal(dist, 0)
    
    try:
        cl = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=100 - threshold,
            metric='precomputed',
            linkage='average'
        )
        return cl.fit_predict(dist)
    except:
        return [0] * n


def cluster_color(cid):
    if cid < 0:
        return "#B0B0B0"
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
    return colors[cid % len(colors)]


# =============================================================================
# RECONSTRUCTION - NEW EDGE-BASED ALGORITHM
# =============================================================================

def extract_edge_signature(contour, num_points=64):
    """Extract edge signature for matching"""
    if contour is None or len(contour) < 5:
        return None
    
    # Resample contour to fixed number of points
    contour = contour.reshape(-1, 2).astype(float)
    
    # Calculate cumulative arc length
    diffs = np.diff(contour, axis=0)
    lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(lengths)])
    total_len = cumlen[-1]
    
    if total_len < 1:
        return None
    
    # Resample at uniform arc length
    target_lens = np.linspace(0, total_len, num_points)
    resampled = np.zeros((num_points, 2))
    
    for i, target in enumerate(target_lens):
        idx = np.searchsorted(cumlen, target)
        idx = min(idx, len(cumlen) - 1)
        if idx == 0:
            resampled[i] = contour[0]
        else:
            t = (target - cumlen[idx - 1]) / (cumlen[idx] - cumlen[idx - 1] + 1e-6)
            resampled[i] = contour[idx - 1] + t * (contour[idx] - contour[idx - 1])
    
    # Calculate curvature at each point
    curvature = np.zeros(num_points)
    for i in range(num_points):
        p_prev = resampled[(i - 2) % num_points]
        p_curr = resampled[i]
        p_next = resampled[(i + 2) % num_points]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        
        curvature[i] = np.arctan2(cross, dot + 1e-6)
    
    return {
        'points': resampled,
        'curvature': curvature,
        'length': total_len
    }


def find_edge_matches(pieces, min_match_score=0.3):
    """Find matching edges between pieces"""
    n = len(pieces)
    matches = []
    
    # Extract edge signatures
    signatures = []
    for p in pieces:
        contour = p.get('contour_global', p.get('contour'))
        sig = extract_edge_signature(contour)
        signatures.append(sig)
    
    # Compare all pairs
    for i in range(n):
        if signatures[i] is None:
            continue
        
        for j in range(i + 1, n):
            if signatures[j] is None:
                continue
            
            # Try to match edge segments
            best_score = 0
            best_rotation = 0
            best_offset = (0, 0)
            
            curv_i = signatures[i]['curvature']
            curv_j = signatures[j]['curvature']
            pts_i = signatures[i]['points']
            pts_j = signatures[j]['points']
            
            # Try different rotations of piece j
            num_pts = len(curv_i)
            for rot in range(0, num_pts, num_pts // 8):
                # Rotate and flip curvature (matching edges have opposite curvature)
                curv_j_rot = np.roll(-curv_j[::-1], rot)
                
                # Find best alignment
                correlation = np.correlate(curv_i, curv_j_rot, mode='same')
                max_corr = np.max(correlation)
                
                if max_corr > best_score:
                    best_score = max_corr
                    best_rotation = rot
            
            # Normalize score
            score = best_score / (num_pts * np.pi)
            
            # Add color similarity bonus
            if 'features' in pieces[i] and 'features' in pieces[j]:
                c1, c2 = pieces[i]['features']['color'], pieces[j]['features']['color']
                color_dist = np.linalg.norm(c1 - c2)
                color_bonus = max(0, 0.3 - color_dist / 500)
                score += color_bonus
            
            if score > min_match_score:
                matches.append({
                    'piece_i': i,
                    'piece_j': j,
                    'score': score * 100,
                    'rotation': best_rotation * 360 / num_pts
                })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches


def reconstruct_group(pieces, canvas_size=800):
    """
    TELLER-REKONSTRUKTION
    
    Für dekorierte Teller/Keramik:
    1. Trennt Randstücke (mit Muster) von Mittelstücken
    2. Nutzt pattern_direction um die Orientierung zu bestimmen
    3. Verteilt Randstücke gleichmäßig im Kreis
    4. Platziert Mittelstücke im Zentrum
    
    Das farbige Muster (Blau/Grün) zeigt immer nach AUẞEN!
    """
    if not pieces or len(pieces) == 0:
        return None, [], []
    
    n = len(pieces)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 242
    center = canvas_size // 2
    
    # Trennen: Stücke mit Muster (Rand) vs. ohne (Mitte)
    edge_pieces = []
    center_pieces = []
    
    for i, p in enumerate(pieces):
        # Prüfe ob Stück ein farbiges Muster hat
        if p.get('has_pattern') or p.get('is_edge_piece'):
            edge_pieces.append((i, p))
        else:
            center_pieces.append((i, p))
    
    # Wenn keine Rand-Stücke erkannt, alle als Rand behandeln
    if not edge_pieces:
        edge_pieces = [(i, p) for i, p in enumerate(pieces)]
        center_pieces = []
    
    matches = []
    placements = [{
        'x': center,
        'y': center,
        'rotation': 0,
        'scale': 1.0,
        'placed': False
    } for _ in range(n)]
    
    # Sortiere Randstücke nach ihrer Muster-Richtung
    edge_with_angles = []
    for idx, piece in edge_pieces:
        # Nutze pattern_direction wenn verfügbar
        if piece.get('pattern_direction') is not None:
            angle = piece['pattern_direction']
        else:
            # Fallback: Nutze Schwerpunkt relativ zum Bildzentrum
            cx_p, cy_p = piece['centroid']
            angle = np.arctan2(cy_p - center, cx_p - center)
        
        edge_with_angles.append((idx, piece, angle))
    
    # Sortiere nach Winkel
    edge_with_angles.sort(key=lambda x: x[2])
    
    # Platziere Randstücke im Kreis
    n_edge = len(edge_with_angles)
    plate_radius = canvas_size * 0.38  # Radius für Rand-Platzierung
    scale = 0.45  # Skalierungsfaktor
    
    for i, (idx, piece, orig_angle) in enumerate(edge_with_angles):
        # Verteile gleichmäßig um den Kreis
        target_angle = 2 * np.pi * i / n_edge if n_edge > 0 else 0
        
        # Rotation: Das Muster soll nach außen zeigen
        # orig_angle ist die aktuelle Muster-Richtung
        # target_angle ist wo das Muster hin soll
        rotation = np.degrees(target_angle - orig_angle)
        
        # Position
        px = center + plate_radius * np.cos(target_angle)
        py = center + plate_radius * np.sin(target_angle)
        
        placements[idx] = {
            'x': px,
            'y': py,
            'rotation': rotation,
            'scale': scale,
            'placed': True
        }
        
        # Verbindungen zwischen benachbarten Stücken
        if i > 0:
            prev_idx = edge_with_angles[i-1][0]
            matches.append({
                'piece_i': prev_idx,
                'piece_j': idx,
                'score': 85.0
            })
    
    # Kreis schließen
    if n_edge > 2:
        matches.append({
            'piece_i': edge_with_angles[-1][0],
            'piece_j': edge_with_angles[0][0],
            'score': 85.0
        })
    
    # Platziere Mittelstücke im Zentrum
    n_center = len(center_pieces)
    inner_radius = canvas_size * 0.08
    
    for i, (idx, piece) in enumerate(center_pieces):
        if n_center == 1:
            px, py = center, center
        else:
            angle = 2 * np.pi * i / n_center
            dist = inner_radius + (i % 2) * 20
            px = center + dist * np.cos(angle)
            py = center + dist * np.sin(angle)
        
        placements[idx] = {
            'x': px,
            'y': py,
            'rotation': 0,
            'scale': scale,
            'placed': True
        }
    
    # Draw pieces on canvas
    for idx, piece in enumerate(pieces):
        if 'thumbnail' not in piece:
            continue
        
        try:
            thumb = piece['thumbnail'].copy()
            mask = piece.get('mask', np.ones(thumb.shape[:2], dtype=np.uint8) * 255)
            
            h, w = thumb.shape[:2]
            max_dim = max(h, w)
            scale = min(100 / max_dim, 1.5) * placements[idx].get('scale', 1.0)
            
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            
            thumb_s = cv2.resize(thumb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # Rotate
            rot_deg = placements[idx].get('rotation', 0)
            if abs(rot_deg) > 0.5:
                M = cv2.getRotationMatrix2D((nw // 2, nh // 2), rot_deg, 1.0)
                cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
                nw_r = int(nh * sin_v + nw * cos_v)
                nh_r = int(nh * cos_v + nw * sin_v)
                M[0, 2] += (nw_r - nw) / 2
                M[1, 2] += (nh_r - nh) / 2
                
                thumb_s = cv2.warpAffine(thumb_s, M, (nw_r, nh_r), borderValue=(245, 245, 245))
                mask_s = cv2.warpAffine(mask_s, M, (nw_r, nh_r), borderValue=0)
                nw, nh = nw_r, nh_r
            
            # Position
            px = int(placements[idx]['x']) - nw // 2
            py = int(placements[idx]['y']) - nh // 2
            
            # Clip to canvas
            x1 = max(0, px)
            y1 = max(0, py)
            x2 = min(canvas_size, px + nw)
            y2 = min(canvas_size, py + nh)
            
            sx1 = x1 - px
            sy1 = y1 - py
            
            if x2 > x1 and y2 > y1:
                roi = canvas[y1:y2, x1:x2]
                t_roi = thumb_s[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
                m_roi = mask_s[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
                
                if roi.shape[:2] == t_roi.shape[:2]:
                    mask_3ch = cv2.cvtColor(m_roi, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                    blended = (t_roi.astype(float) * mask_3ch + roi.astype(float) * (1 - mask_3ch))
                    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        except Exception as e:
            continue
    
    return canvas, placements, matches


# =============================================================================
# PDF LABELS - FIXED LAYOUT
# =============================================================================

def create_labels_pdf(pieces, username):
    """Create PDF with proper layout"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    page_w, page_h = A4
    
    # Layout settings
    margin_left = 15 * mm
    margin_top = 20 * mm
    margin_right = 15 * mm
    margin_bottom = 15 * mm
    
    cols = 3
    rows = 4
    
    usable_w = page_w - margin_left - margin_right
    usable_h = page_h - margin_top - margin_bottom
    
    cell_w = usable_w / cols
    cell_h = usable_h / rows
    
    qr_size = 18 * mm
    thumb_size = 12 * mm
    
    # Header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, page_h - 12 * mm, f"ShardMind - {username}")
    c.setFont("Helvetica", 8)
    c.drawString(margin_left, page_h - 16 * mm, f"{datetime.now().strftime('%Y-%m-%d')} | {len(pieces)} Etiketten")
    
    items_per_page = cols * rows
    
    for idx, piece in enumerate(pieces):
        # New page if needed
        page_idx = idx // items_per_page
        if idx > 0 and idx % items_per_page == 0:
            c.showPage()
            # Header on new page
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_left, page_h - 12 * mm, f"ShardMind - {username}")
            c.setFont("Helvetica", 8)
            c.drawString(margin_left, page_h - 16 * mm, f"Seite {page_idx + 1}")
        
        # Position in grid
        pos_on_page = idx % items_per_page
        col = pos_on_page % cols
        row = pos_on_page // cols
        
        # Cell position (top-left corner)
        cell_x = margin_left + col * cell_w
        cell_y = page_h - margin_top - (row + 1) * cell_h
        
        # Padding inside cell
        pad = 3 * mm
        
        # QR Code
        piece_id = piece.get('id', gen_id())
        url = f"{BASE_URL}/?piece={piece_id}"
        
        try:
            qr = gen_qr(url, 4)
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format='PNG')
            qr_buf.seek(0)
            c.drawImage(ImageReader(qr_buf), cell_x + pad, cell_y + cell_h - pad - qr_size, 
                       width=qr_size, height=qr_size)
        except:
            pass
        
        # Thumbnail
        if 'thumbnail' in piece:
            try:
                thumb_rgb = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(thumb_rgb)
                pil.thumbnail((100, 100))
                tb = io.BytesIO()
                pil.save(tb, format='PNG')
                tb.seek(0)
                c.drawImage(ImageReader(tb), cell_x + pad + qr_size + 2*mm, 
                           cell_y + cell_h - pad - thumb_size, 
                           width=thumb_size, height=thumb_size)
            except:
                pass
        
        # Text
        text_x = cell_x + pad
        text_y = cell_y + cell_h - pad - qr_size - 4*mm
        
        c.setFont("Helvetica-Bold", 7)
        c.drawString(text_x, text_y, piece_id[:16])
        
        c.setFont("Helvetica", 6)
        text_y -= 3 * mm
        name = piece.get('name', '-')[:25]
        c.drawString(text_x, text_y, name)
        
        text_y -= 2.5 * mm
        proj = piece.get('excavation', '-')[:25]
        c.drawString(text_x, text_y, proj)
        
        # Cell border (dashed)
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setDash(2, 2)
        c.rect(cell_x, cell_y, cell_w, cell_h)
        c.setDash()
    
    # Footer
    c.setFont("Helvetica", 6)
    c.drawString(margin_left, 8 * mm, f"ShardMind v{APP_VERSION} | {BASE_URL}")
    
    c.save()
    buffer.seek(0)
    return buffer


# =============================================================================
# STREAMLIT APP
# =============================================================================

def login_page():
    st.markdown(f"# 🏺 {t('app_title')}")
    st.caption(t('app_tagline'))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        lang = st.selectbox("🌍", ['de', 'en'], 
                           format_func=lambda x: '🇩🇪 Deutsch' if x == 'de' else '🇬🇧 English',
                           index=0 if st.session_state.get('language', 'de') == 'de' else 1)
        if lang != st.session_state.get('language', 'de'):
            st.session_state.language = lang
            st.rerun()
    
    tab1, tab2 = st.tabs([t('login'), t('register')])
    
    with tab1:
        with st.form("login"):
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
        with st.form("register"):
            new_user = st.text_input(t('username'), key='ru')
            new_pw = st.text_input(t('password'), type='password', key='rp')
            confirm = st.text_input(t('password_confirm'), type='password')
            st.info(f"ℹ️ {t('account_info')}")
            
            if st.form_submit_button(t('register_btn'), use_container_width=True):
                if new_pw != confirm:
                    st.error(t('register_error_password'))
                else:
                    ok, msg = register_user(new_user, new_pw)
                    if ok:
                        st.success(t(msg))
                    else:
                        st.error(t(msg))


def main():
    st.set_page_config(page_title=f"ShardMind v{APP_VERSION}", page_icon="🏺", layout="wide")
    
    # Init session state
    defaults = {
        'language': 'de',
        'logged_in': False,
        'username': '',
        'pieces': [],
        'cluster_names': {},
        'custom_labels': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    if not st.session_state.logged_in:
        login_page()
        return
    
    username = st.session_state.username
    user_data = get_user_data(username)
    
    # URL params
    search_id = st.query_params.get('piece', None)
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown(f"## 🏺 {t('app_title')}")
        st.caption(f"v{APP_VERSION}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"👤 {username}")
        with col2:
            if st.button(t('logout_btn')):
                st.session_state.logged_in = False
                st.session_state.pieces = []
                st.rerun()
        
        lang = st.selectbox(t('language'), ['de', 'en'],
                           format_func=lambda x: '🇩🇪 Deutsch' if x == 'de' else '🇬🇧 English',
                           index=0 if st.session_state.language == 'de' else 1, key='lang_m')
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
        
        st.divider()
        
        files = st.file_uploader(t('upload_photos'), type=['png', 'jpg', 'jpeg'], 
                                 accept_multiple_files=True, help=t('upload_hint'))
        
        min_area = st.slider(t('min_size'), 50, 1000, 150)
        threshold = st.slider(t('cluster_sens'), 10, 80, 40)
        
        mode = st.selectbox(t('detection_mode'), ['porcelain', 'auto', 'light_bg', 'dark_bg', 'high_contrast'],
                           format_func=lambda x: {
                               'auto': t('mode_auto'),
                               'light_bg': t('mode_light_bg'),
                               'dark_bg': t('mode_dark_bg'),
                               'high_contrast': t('mode_high_contrast'),
                               'porcelain': t('mode_porcelain')
                           }[x])
        
        project = st.text_input(t('project'), value=f"Project_{datetime.now().strftime('%Y')}")
        
        st.divider()
        
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
            if 'recon_image' in st.session_state:
                del st.session_state.recon_image
            st.rerun()
        
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric(t('fragments'), len(user_data.get('pieces', {})))
        col2.metric(t('groups'), len(user_data.get('clusters', {})))
    
    # === PROCESS ===
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
    
    if active:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏺 {t('fragments')}", len(active))
        c2.metric(f"📦 {t('groups')}", n_clusters)
        c3.metric("🗺️", project[:20])
    
    # === TABS ===
    tabs = st.tabs([t('tab_start'), t('tab_gallery'), t('tab_groups'), 
                    t('tab_reconstruction'), t('tab_database'), t('tab_labels'), t('tab_help')])
    
    # TAB 0: START
    with tabs[0]:
        st.header(f"🏺 {t('app_title')}")
        st.write(t('app_tagline'))
        
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
                c1, c2 = st.columns([1, 2])
                with c1:
                    if 'thumbnail' in found:
                        st.image(cv2.cvtColor(found['thumbnail'], cv2.COLOR_BGR2RGB))
                with c2:
                    st.write(f"**ID:** `{found['id']}`")
                    st.write(f"**Name:** {found.get('name', '-')}")
                    st.write(f"**Material:** {found.get('material', '-')}")
            else:
                st.warning(f"{t('search_not_found')}: {search_id}")
            st.divider()
        
        # Search
        st.subheader(t('search_title'))
        c1, c2 = st.columns([3, 1])
        with c1:
            search_input = st.text_input("ID", placeholder="SM-XXXXXXXX", label_visibility="collapsed")
        with c2:
            if st.button(t('search_btn'), use_container_width=True):
                if search_input:
                    st.query_params['piece'] = search_input.strip().upper()
                    st.rerun()
        
        st.divider()
        
        # Demo
        st.subheader(t('demo_title'))
        st.write(t('demo_hint'))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🏺 {t('demo_pottery')}**")
            demo1 = generate_demo_pottery(6)
            st.image(cv2.cvtColor(demo1, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.download_button(t('demo_download'), get_demo_bytes(demo1), "demo_pottery.png", "image/png", 
                             use_container_width=True, key="d1")
        
        with c2:
            st.markdown(f"**🍽️ {t('demo_plate')}**")
            demo2 = generate_demo_plate(5)
            st.image(cv2.cvtColor(demo2, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.download_button(t('demo_download'), get_demo_bytes(demo2), "demo_plate.png", "image/png",
                             use_container_width=True, key="d2")
    
    # TAB 1: GALLERY
    with tabs[1]:
        st.header(t('detected_fragments'))
        if active:
            cols = st.columns(5)
            for i, p in enumerate(active):
                with cols[i % 5]:
                    thumb_rgb = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB)
                    color_hex = cluster_color(p.get('cluster', -1))
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    bordered = cv2.copyMakeBorder(thumb_rgb, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(r, g, b))
                    st.image(bordered, use_container_width=True)
                    st.caption(f"**{p['id'][:11]}**\n{p.get('name', '')[:12]}")
        else:
            st.info(t('upload_first'))
    
    # TAB 2: GROUPS
    with tabs[2]:
        st.header(t('groups_title'))
        if active:
            sorted_c = sorted([c for c in cluster_ids if c >= 0])
            if not sorted_c:
                st.warning(t('no_groups'))
            else:
                for cid in sorted_c:
                    cp = [p for p in active if p.get('cluster') == cid]
                    mats = [p.get('material', 'Unknown') for p in cp]
                    common = max(set(mats), key=mats.count) if mats else 'Unknown'
                    default_name = st.session_state.cluster_names.get(cid, f"{common}_Gruppe_{cid + 1}")
                    
                    with st.expander(f"📦 {default_name} ({len(cp)} {t('pieces')})", expanded=True):
                        name = st.text_input(t('group_name'), value=default_name, key=f"gn_{cid}")
                        st.session_state.cluster_names[cid] = name
                        
                        pcols = st.columns(min(6, len(cp)))
                        for i, p in enumerate(cp[:6]):
                            with pcols[i]:
                                st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.caption(p['id'][:8])
                        
                        if st.button(t('save_btn'), key=f"sv_{cid}"):
                            key = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            user_data['clusters'][key] = {
                                'name': name,
                                'created': datetime.now().isoformat(),
                                'piece_ids': [p['id'] for p in cp]
                            }
                            for p in cp:
                                pc = {k: v for k, v in p.items() if k not in ['features', 'contour_global']}
                                user_data['pieces'][p['id']] = pc
                            save_user_data(username, user_data)
                            st.success(t('saved'))
                            st.rerun()
        else:
            st.info(t('upload_first'))
    
    # TAB 3: RECONSTRUCTION
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
            
            st.write(f"**{len(rp)} {t('pieces')}**")
            pcols = st.columns(min(8, len(rp)))
            for i, p in enumerate(rp[:8]):
                with pcols[i]:
                    st.image(cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            canvas_size = st.slider(t('canvas_size'), 500, 1200, 800, 50)
            
            if st.button(t('calculate_btn'), type='primary', use_container_width=True):
                with st.spinner(t('analyzing')):
                    img, plc, matches = reconstruct_group(rp, canvas_size)
                    if img is not None:
                        st.session_state.recon_image = img
                        st.session_state.recon_matches = matches
                        st.success(f"✓ {len(matches)} {t('matches_found')}")
            
            if 'recon_image' in st.session_state and st.session_state.recon_image is not None:
                c1, c2 = st.columns([2, 1])
                with c1:
                    rgb = cv2.cvtColor(st.session_state.recon_image, cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                    
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format='PNG')
                    st.download_button(t('export_btn'), buf.getvalue(), "reconstruction.png", "image/png", 
                                      use_container_width=True)
                
                with c2:
                    st.markdown(f"**{t('edge_matches')}**")
                    for m in st.session_state.get('recon_matches', [])[:8]:
                        st.write(f"• Teil {m['piece_i']+1} ↔ {m['piece_j']+1}: {m['score']:.0f}%")
        else:
            st.info(t('upload_first'))
    
    # TAB 4: DATABASE
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
    
    # TAB 5: LABELS
    with tabs[5]:
        st.header(t('labels_title'))
        
        source = st.radio(t('label_source'), 
                         [t('label_session'), t('label_database'), t('label_custom')],
                         horizontal=True)
        
        label_pieces = []
        
        if source == t('label_session'):
            label_pieces = active
        elif source == t('label_database'):
            label_pieces = list(user_data.get('pieces', {}).values())
        else:
            st.subheader(t('label_custom'))
            
            with st.form("custom_form"):
                c1, c2 = st.columns(2)
                with c1:
                    cid = st.text_input("ID", placeholder="SM-CUSTOM01")
                    cname = st.text_input(t('custom_label_name'))
                with c2:
                    cdesc = st.text_input(t('custom_label_desc'))
                    cproj = st.text_input(t('project'), value=project)
                
                if st.form_submit_button(t('add_custom_label'), use_container_width=True):
                    new_lbl = {
                        'id': cid if cid else gen_id(),
                        'name': cname or "Custom",
                        'description': cdesc,
                        'excavation': cproj
                    }
                    st.session_state.custom_labels.append(new_lbl)
                    st.success(t('saved'))
                    st.rerun()
            
            if st.session_state.custom_labels:
                st.write(f"**{t('custom_labels_list')}:**")
                for lbl in st.session_state.custom_labels:
                    st.write(f"• {lbl['id']} - {lbl['name']}")
                
                if st.button(t('clear_custom')):
                    st.session_state.custom_labels = []
                    st.rerun()
                
                label_pieces = st.session_state.custom_labels
        
        if label_pieces:
            st.write(f"**{len(label_pieces)} Labels**")
            
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
                    st.download_button("📥 PDF", pdf.getvalue(),
                                      f"labels_{datetime.now().strftime('%Y%m%d')}.pdf",
                                      "application/pdf", use_container_width=True)
        else:
            st.info(t('upload_first'))
    
    # TAB 6: HELP
    with tabs[6]:
        st.header(t('help_title'))
        st.markdown(HELP_DE if st.session_state.language == 'de' else HELP_EN)


if __name__ == "__main__":
    main()
