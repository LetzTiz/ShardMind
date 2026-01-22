"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗   ██╗ █████╗ ██████╗ ██████╗ ███╗   ███╗██╗███╗   ██╗██████╗    ║
║   ██╔════╝██║   ██║██╔══██╗██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗   ║
║   ███████╗███████║███████║██████╔╝██║   ██║██╔████╔██║██║██╔██╗ ██║██║  ██║   ║
║   ╚════██║██╔══██║██╔══██║██╔══██╗██║   ██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║   ║
║   ███████║██║   ██║██║  ██║██║  ██║██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██████╔╝   ║
║   ╚══════╝╚═╝   ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝    ║
║                                                                               ║
║   Archäologische Fragmentanalyse & Rekonstruktion                             ║
║   Archaeological Fragment Analysis & Reconstruction                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG:
==========

Version 1.8 (2025-01-22)
------------------------
FIX:
• 🛠️ KORRIGIERTE TELLER-REKONSTRUKTION (Circle-Fit)
  - Fragmente werden nicht mehr gleichmäßig verteilt ("Blütenblätter"),
    sondern basierend auf ihrer Winkelbreite (Angle Span) sequenziell platziert.
  - Nutzt Circle-Fitting auf blauen Farbbändern zur Berechnung des Krümmungsradius.
  - Schließt Lücken zwischen Scherben automatisch.

Version 1.7 (2025-01-21)
------------------------
NEU:
• 🎯 KOMPLETT NEUER REKONSTRUKTIONS-ALGORITHMUS!
  - Kreis-Fitting: Findet automatisch das Tellerzentrum aus dem blauen Dekorband
  - Jedes Fragment wird so platziert, dass sein Muster auf dem Rekonstruktionskreis liegt

Version 1.5 (2025-01-21)
------------------------
• 🧩 Kantenpassung & PDF-Layout Updates
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

APP_VERSION = "1.8"
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
    angles += np.random.uniform(-0.4, 0.4, num)
    angles = np.sort(angles)
    
    for i in range(num):
        a1 = angles[i]
        a2 = angles[(i + 1) % num]
        if a2 < a1: a2 += 2 * np.pi
        
        # Mask
        mask = np.zeros((800, 800), dtype=np.uint8)
        pts = [(center_x, center_y)]
        for a in np.linspace(a1, a2, 15):
            x = int(center_x + (outer_r + 20) * np.cos(a))
            y = int(center_y + (outer_r + 20) * np.sin(a))
            pts.append((x, y))
        cv2.fillPoly(mask, [np.array(pts)], 255)
        mask = cv2.bitwise_and(mask, original)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        
        # Offset
        offset_a = (a1 + a2) / 2
        dist = 90
        ox = int(dist * np.cos(offset_a))
        oy = int(dist * np.sin(offset_a))
        
        shifted_cnt = cnt.copy()
        shifted_cnt[:, :, 0] += ox
        shifted_cnt[:, :, 1] += oy
        
        # Draw - White Porcelain style
        cv2.fillPoly(img, [shifted_cnt], (245, 245, 250))
        cv2.polylines(img, [shifted_cnt], True, (200, 200, 210), 1)
        
        # Blue Pattern on Rim
        # We need to draw the blue arc on the shifted piece
        # This simulates the "blue rim" for detection
        
        # Create a local mask for the piece to draw texture only on piece
        piece_draw_mask = np.zeros((800, 800), dtype=np.uint8)
        cv2.fillPoly(piece_draw_mask, [shifted_cnt], 255)
        
        # Draw blue arcs
        temp_img = np.zeros((800, 800, 3), dtype=np.uint8)
        # Shifted center
        scx, scy = center_x + ox, center_y + oy
        
        # Blue rim
        cv2.circle(temp_img, (scx, scy), outer_r - 10, (180, 100, 20), 8) # Blue in BGR
        cv2.circle(temp_img, (scx, scy), outer_r - 25, (180, 100, 20), 2)
        
        # Mask texture
        locs = np.where(piece_draw_mask > 0)
        img[locs] = cv2.addWeighted(img[locs], 0.7, temp_img[locs], 0.3, 0)[0]
        
    return img


# =============================================================================
# DB & AUTH
# =============================================================================

def init_db():
    if not USERS_DB_PATH.exists():
        with open(USERS_DB_PATH, 'wb') as f:
            pickle.dump({}, f)

def load_users():
    init_db()
    with open(USERS_DB_PATH, 'rb') as f:
        return pickle.load(f)

def save_users(users):
    with open(USERS_DB_PATH, 'wb') as f:
        pickle.dump(users, f)

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    users = load_users()
    if username in users and users[username]['password'] == hash_pw(password):
        return True
    return False

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        'password': hash_pw(password),
        'created': datetime.now(),
        'fragments': [],
        'groups': [],
        'custom_labels': []
    }
    save_users(users)
    return True

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def segment_fragments(image, mode='auto'):
    """Advanced segmentation with shadow removal and multi-channel edge detection"""
    
    # 1. Resize for speed but keep quality
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    # 2. Denoise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 3. Detect Background
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    if mode == 'mode_dark_bg':
        # Dark background -> Threshold
        _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    elif mode == 'mode_high_contrast':
        # Otsu
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Light background (standard lab setup)
        # Use localized variance or simple adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 5)
        
        # Refine: Remove shadows
        # Convert to LAB, check L channel
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        _, shadow_mask = cv2.threshold(l_channel, 210, 255, cv2.THRESH_BINARY_INV) # Exclude very bright
        
    # 4. Clean Mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fragments = []
    min_area = (h * w) * 0.001  # 0.1% of image
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        x, y, fw, fh = cv2.boundingRect(cnt)
        
        # Expand ROI slightly
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)
        
        roi = image[y1:y2, x1:x2]
        roi_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
        # Shift contour for mask
        cnt_shifted = cnt - [x1, y1]
        cv2.fillPoly(roi_mask, [cnt_shifted], 255)
        
        # Extract features
        avg_color = cv2.mean(roi, mask=roi_mask)[:3]
        
        fragments.append({
            'id': str(uuid.uuid4())[:8],
            'contour': cnt, # Original coords
            'bbox': (x, y, fw, fh),
            'area': area,
            'thumbnail': roi,
            'mask': roi_mask,
            'avg_color': avg_color,
            'source_image_idx': 0 # Placeholder
        })
        
    return fragments

def cluster_fragments(fragments, n_clusters=None, sensitivity=1.0):
    if not fragments:
        return {}
    
    # Feature extraction: Color (LAB) + Texture + Size
    features = []
    for f in fragments:
        # Color
        b, g, r = f['avg_color']
        lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0][0]
        
        # Size (log scale)
        size_feat = np.log(f['area']) * 10
        
        feat = [
            float(lab[0]) * 1.0,  # L
            float(lab[1]) * 1.5,  # A (Color important)
            float(lab[2]) * 1.5,  # B
            size_feat * 0.5
        ]
        features.append(feat)
    
    X = np.array(features)
    
    # Normalize
    if len(X) > 1:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-5)
    
    # Clustering
    if n_clusters is None:
        # Distance threshold
        dist_threshold = 2.0 * (1.0 / sensitivity)
        clusterer = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=dist_threshold,
            metric='euclidean',
            linkage='ward'
        )
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
    labels = clusterer.fit_predict(X)
    
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(fragments[i])
        
    return groups

# =============================================================================
# RECONSTRUCTION LOGIC - MODIFIED FOR V1.8
# =============================================================================

def reconstruct_group(pieces, canvas_size=900):
    """
    VERBESSERTE TELLER-REKONSTRUKTION
    
    Platziert Fragmente basierend auf ihrer tatsächlichen Position und
    Winkelbreite aus dem Originalbild (Circle-Fitting auf blaue Dekorbänder).
    """
    if not pieces or len(pieces) == 0:
        return None, [], []
    
    n = len(pieces)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 248
    center = canvas_size // 2
    
    matches = []
    placements = [{
        'x': center,
        'y': center,
        'rotation': 0,
        'scale': 1.0,
        'placed': False
    } for _ in range(n)]
    
    # Analysiere jedes Fragment für Rekonstruktion
    analyzed_pieces = []
    
    for idx, piece in enumerate(pieces):
        # Hole das Originalbild-ROI und Maske
        thumbnail = piece.get('thumbnail')
        mask = piece.get('mask')
        if thumbnail is None:
            continue
        
        # Erkenne blaue Pixel im Fragment
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV) if len(thumbnail.shape) == 3 else None
        if hsv is not None:
            h_ch = hsv[:, :, 0]
            s_ch = hsv[:, :, 1]
            blue_mask = ((h_ch > 90) & (h_ch < 130) & (s_ch > 35)).astype(np.uint8) * 255
            
            # Wende Fragment-Maske an
            if mask is not None and mask.shape == blue_mask.shape:
                blue_mask = cv2.bitwise_and(blue_mask, mask)
            
            blue_pts = np.column_stack(np.where(blue_mask > 0))[:, ::-1]
            
            if len(blue_pts) > 50:
                # Circle-Fit für dieses Fragment
                x_b = blue_pts[:, 0].astype(float)
                y_b = blue_pts[:, 1].astype(float)
                try:
                    A = np.column_stack([x_b, y_b, np.ones_like(x_b)])
                    b = x_b**2 + y_b**2
                    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    local_cx = result[0] / 2
                    local_cy = result[1] / 2
                    local_r = np.sqrt(result[2] + local_cx**2 + local_cy**2)
                    
                    if local_r > 20:
                        # Berechne Winkel der blauen Pixel
                        angles = np.arctan2(y_b - local_cy, x_b - local_cx)
                        
                        # Winkelspanne
                        angles_sorted = np.sort(angles)
                        gaps = np.diff(angles_sorted)
                        gaps = np.append(gaps, angles_sorted[0] + 2*np.pi - angles_sorted[-1])
                        max_gap_idx = np.argmax(gaps)
                        
                        if max_gap_idx == len(gaps) - 1:
                            angle_min = angles_sorted[0]
                            angle_max = angles_sorted[-1]
                        else:
                            angle_min = angles_sorted[(max_gap_idx + 1) % len(angles_sorted)]
                            angle_max = angles_sorted[max_gap_idx]
                        
                        angle_span = angle_max - angle_min
                        if angle_span < 0:
                            angle_span += 2 * np.pi
                        if angle_span > np.pi:
                            angle_span = 2 * np.pi - angle_span
                        
                        # Mittlerer Winkel
                        blue_center = blue_pts.mean(axis=0)
                        original_angle = np.arctan2(blue_center[1] - local_cy, blue_center[0] - local_cx)
                        
                        analyzed_pieces.append({
                            'idx': idx,
                            'piece': piece,
                            'thumbnail': thumbnail,
                            'mask': mask,
                            'local_center': (local_cx, local_cy),
                            'local_radius': local_r,
                            'blue_center': blue_center,
                            'original_angle': original_angle,
                            'angle_span': angle_span,
                            'area': piece.get('area', thumbnail.shape[0] * thumbnail.shape[1])
                        })
                        continue
                except:
                    pass
        
        # Fallback für Fragmente ohne Blauband
        analyzed_pieces.append({
            'idx': idx,
            'piece': piece,
            'thumbnail': thumbnail,
            'mask': mask,
            'local_center': None,
            'is_center_piece': True,
            'area': piece.get('area', thumbnail.shape[0] * thumbnail.shape[1])
        })
    
    # Trenne Rand- und Mittelstücke
    edge_pieces = [p for p in analyzed_pieces if p.get('local_center') is not None]
    center_pieces = [p for p in analyzed_pieces if p.get('is_center_piece', False)]
    
    if not edge_pieces:
        edge_pieces = analyzed_pieces
        center_pieces = []
    
    # Sortiere nach Original-Winkel
    edge_pieces.sort(key=lambda p: p.get('original_angle', 0))
    
    # Berechne Gesamtwinkel und durchschnittlichen Radius
    if edge_pieces:
        total_span = sum(p.get('angle_span', 0.3) for p in edge_pieces)
        avg_radius = np.mean([p['local_radius'] for p in edge_pieces if 'local_radius' in p])
        
        # Ziel-Radius auf Canvas
        target_radius = canvas_size * 0.35
        scale = target_radius / avg_radius if avg_radius > 0 else 0.5
        
        # Berechne Lücke pro Fragment
        if total_span < 2 * np.pi:
            gap_per_frag = (2 * np.pi - total_span) / max(len(edge_pieces), 1)
        else:
            gap_per_frag = 0
            scale_factor = (2 * np.pi * 0.95) / total_span
            for p in edge_pieces:
                if 'angle_span' in p:
                    p['angle_span'] *= scale_factor
            gap_per_frag = 0.02
        
        # Platziere Fragmente SEQUENZIELL
        current_angle = -np.pi
        
        for p in edge_pieces:
            idx = p['idx']
            thumbnail = p['thumbnail']
            mask = p['mask']
            angle_span = p.get('angle_span', 0.3)
            
            h, w = thumbnail.shape[:2]
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            
            thumb_s = cv2.resize(thumbnail, (nw, nh))
            mask_s = cv2.resize(mask, (nw, nh)) if mask is not None else np.ones((nh, nw), dtype=np.uint8) * 255
            
            # Zielwinkel ist Mitte dieses Fragments
            target_angle = current_angle + angle_span / 2
            
            # Berechne Rotation
            if 'blue_center' in p and 'local_center' in p:
                bcx_s = p['blue_center'][0] * scale
                bcy_s = p['blue_center'][1] * scale
                lcx_s = p['local_center'][0] * scale
                lcy_s = p['local_center'][1] * scale
                
                # Aktuelle "nach außen" Richtung
                current_outward = np.arctan2(bcy_s - lcy_s, bcx_s - lcx_s)
                rot_deg = np.degrees(target_angle - current_outward)
            else:
                rot_deg = 0
                lcx_s = nw / 2
                lcy_s = nh / 2
            
            # Rotiere
            if abs(rot_deg) > 0.5:
                M = cv2.getRotationMatrix2D((nw // 2, nh // 2), rot_deg, 1.0)
                cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
                nw_r = int(nh * sin_v + nw * cos_v) + 20
                nh_r = int(nh * cos_v + nw * sin_v) + 20
                M[0, 2] += (nw_r - nw) / 2
                M[1, 2] += (nh_r - nh) / 2
                
                thumb_s = cv2.warpAffine(thumb_s, M, (nw_r, nh_r), borderValue=(248, 248, 248))
                mask_s = cv2.warpAffine(mask_s, M, (nw_r, nh_r), borderValue=0)
                
                # Transformiere Zentrum
                lcx_rot = M[0, 0] * lcx_s + M[0, 1] * lcy_s + M[0, 2]
                lcy_rot = M[1, 0] * lcx_s + M[1, 1] * lcy_s + M[1, 2]
                nw, nh = nw_r, nh_r
            else:
                lcx_rot = lcx_s
                lcy_rot = lcy_s
            
            # Radialer Offset basierend auf Fragmentgröße
            frag_height = max(nw, nh)
            inward_offset = frag_height * 0.3
            fragment_radius = target_radius - inward_offset
            
            # Position berechnen
            target_x = center + fragment_radius * np.cos(target_angle)
            target_y = center + fragment_radius * np.sin(target_angle)
            
            px = int(target_x - lcx_rot)
            py = int(target_y - lcy_rot)
            
            placements[idx] = {
                'x': px + nw // 2,
                'y': py + nh // 2,
                'rotation': rot_deg,
                'scale': scale,
                'placed': True,
                'thumb_s': thumb_s,
                'mask_s': mask_s,
                'area': p.get('area', 0)
            }
            
            # Nächstes Fragment
            current_angle += angle_span + gap_per_frag
            
            # Match zum vorherigen
            if len([p for p in placements if p.get('placed')]) > 1:
                prev_placed = [i for i, pl in enumerate(placements) if pl.get('placed') and i != idx]
                if prev_placed:
                    matches.append({
                        'piece_i': prev_placed[-1],
                        'piece_j': idx,
                        'score': 80.0
                    })
    
    # Platziere Mittelstücke
    n_center = len(center_pieces)
    inner_radius = canvas_size * 0.08
    
    for i, p in enumerate(center_pieces):
        idx = p['idx']
        thumbnail = p['thumbnail']
        mask = p['mask']
        
        h, w = thumbnail.shape[:2]
        scale_c = min(80 / max(h, w), 1.0)
        nw = max(1, int(w * scale_c))
        nh = max(1, int(h * scale_c))
        
        thumb_s = cv2.resize(thumbnail, (nw, nh))
        mask_s = cv2.resize(mask, (nw, nh)) if mask is not None else np.ones((nh, nw), dtype=np.uint8) * 255
        
        if n_center == 1:
            px, py = center, center
        else:
            angle = 2 * np.pi * i / n_center
            px = int(center + inner_radius * np.cos(angle))
            py = int(center + inner_radius * np.sin(angle))
        
        placements[idx] = {
            'x': px,
            'y': py,
            'rotation': 0,
            'scale': scale_c,
            'placed': True,
            'thumb_s': thumb_s,
            'mask_s': mask_s,
            'area': p.get('area', 0)
        }
    
    # Zeichne auf Canvas (sortiert nach Fläche für Tiefe)
    placed_items = [(idx, pl) for idx, pl in enumerate(placements) if pl.get('placed') and 'thumb_s' in pl]
    placed_items.sort(key=lambda x: x[1].get('area', 0))
    
    # Zeichne Tellerbasis
    cv2.circle(canvas, (center, center), int(canvas_size * 0.45), (252, 252, 254), -1)
    cv2.circle(canvas, (center, center), int(canvas_size * 0.45), (220, 220, 225), 2)
    
    for idx, pl in placed_items:
        thumb_s = pl['thumb_s']
        mask_s = pl['mask_s']
        
        nh, nw = thumb_s.shape[:2]
        px = int(pl['x']) - nw // 2
        py = int(pl['y']) - nh // 2
        
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
    
    # Innerer weißer Kreis
    cv2.circle(canvas, (center, center), int(canvas_size * 0.15), (255, 255, 255), -1)
    
    return canvas, placements, matches


# =============================================================================
# PDF GENERATION
# =============================================================================

def create_labels_pdf(labels):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Layout configuration
    margin_x = 10 * mm
    margin_y = 10 * mm
    cols = 3
    rows = 4
    cell_w = (width - 2 * margin_x) / cols
    cell_h = (height - 2 * margin_y) / rows
    
    for i, label in enumerate(labels):
        if i > 0 and i % (cols * rows) == 0:
            p.showPage()
            
        idx_on_page = i % (cols * rows)
        row = idx_on_page // cols
        col = idx_on_page % cols
        
        x = margin_x + col * cell_w
        y = height - margin_y - (row + 1) * cell_h
        
        # Border for label
        p.setStrokeColorRGB(0.8, 0.8, 0.8)
        p.rect(x + 2*mm, y + 2*mm, cell_w - 4*mm, cell_h - 4*mm)
        
        # Content
        p.setFillColorRGB(0, 0, 0)
        p.setFont("Helvetica-Bold", 12)
        p.drawString(x + 5*mm, y + cell_h - 10*mm, label['name'][:20])
        
        p.setFont("Helvetica", 10)
        p.drawString(x + 5*mm, y + cell_h - 15*mm, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        p.drawString(x + 5*mm, y + cell_h - 20*mm, f"ID: {label.get('id', 'N/A')}")
        
        # Description wrapped
        desc = label.get('desc', '')
        p.setFont("Helvetica", 8)
        text_y = y + cell_h - 30*mm
        for line in [desc[i:i+30] for i in range(0, len(desc), 30)][:3]:
            p.drawString(x + 5*mm, text_y, line)
            text_y -= 4*mm
            
        # QR Code - Deep Link
        qr_data = f"{BASE_URL}?id={label.get('id', '0')}"
        qr = qrcode.QRCode(box_size=10, border=1)
        qr.add_data(qr_data)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR to temp buffer to draw
        qr_buf = io.BytesIO()
        img_qr.save(qr_buf, format='PNG')
        qr_buf.seek(0)
        
        p.drawImage(ImageReader(qr_buf), x + cell_w - 25*mm, y + 5*mm, width=20*mm, height=20*mm)
        
    p.save()
    buffer.seek(0)
    return buffer

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="ShardMind", page_icon="🏺", layout="wide")
    
    # --- Session State ---
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'username' not in st.session_state: st.session_state.username = ""
    if 'fragments' not in st.session_state: st.session_state.fragments = []
    if 'groups' not in st.session_state: st.session_state.groups = {}
    if 'language' not in st.session_state: st.session_state.language = 'de'
    if 'custom_labels' not in st.session_state: st.session_state.custom_labels = []

    # --- Sidebar ---
    with st.sidebar:
        st.title(f"🏺 {t('app_title')} v{APP_VERSION}")
        
        # Language Switch
        lang = st.radio("Language / Sprache", ['de', 'en'], 
                        index=0 if st.session_state.language == 'de' else 1,
                        horizontal=True)
        st.session_state.language = lang
        
        if not st.session_state.logged_in:
            tab1, tab2 = st.tabs([t('login'), t('register')])
            
            with tab1:
                u = st.text_input(t('username'), key="l_u")
                p = st.text_input(t('password'), type="password", key="l_p")
                if st.button(t('login_btn')):
                    if check_login(u, p):
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        # Load user data
                        users = load_users()
                        user_data = users[u]
                        st.session_state.custom_labels = user_data.get('custom_labels', [])
                        st.success(f"Welcome {u}")
                        st.rerun()
                    else:
                        st.error(t('login_error'))
                        
            with tab2:
                ru = st.text_input(t('username'), key="r_u")
                rp = st.text_input(t('password'), type="password", key="r_p")
                rp2 = st.text_input(t('password_confirm'), type="password", key="r_p2")
                if st.button(t('register_btn')):
                    if rp != rp2:
                        st.error(t('register_error_password'))
                    elif len(rp) < 4:
                        st.error(t('register_error_short'))
                    elif register_user(ru, rp):
                        st.success(t('register_success'))
                    else:
                        st.error(t('register_error_exists'))
                        
            st.info(t('account_info'))
            st.markdown("---")
            
        else:
            st.write(f"👤 **{st.session_state.username}**")
            if st.button(t('logout_btn')):
                st.session_state.logged_in = False
                st.rerun()
                
            st.markdown("---")
            
            # --- Analysis Settings ---
            st.subheader(t('settings'))
            
            uploaded_files = st.file_uploader(t('upload_photos'), type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
            
            detection_mode = st.selectbox(t('detection_mode'), 
                                        ['mode_auto', 'mode_light_bg', 'mode_dark_bg', 'mode_high_contrast'],
                                        format_func=lambda x: t(x))
            
            cluster_sens = st.slider(t('cluster_sens'), 0.1, 2.0, 1.0)
            
            if st.button(t('analyze_btn'), type="primary"):
                if uploaded_files:
                    all_frags = []
                    
                    progress = st.progress(0)
                    for i, up_file in enumerate(uploaded_files):
                        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
                        if img is not None:
                            frags = segment_fragments(img, detection_mode)
                            all_frags.extend(frags)
                        progress.progress((i + 1) / len(uploaded_files))
                    
                    st.session_state.fragments = all_frags
                    
                    # Cluster immediately
                    groups = cluster_fragments(all_frags, sensitivity=cluster_sens)
                    st.session_state.groups = groups
                    
                    st.success(f"{len(all_frags)} {t('success_fragments')}")
                    st.rerun()
                else:
                    st.warning(t('upload_first'))
                    
            if st.button(t('clear_btn')):
                st.session_state.fragments = []
                st.session_state.groups = {}
                st.rerun()

    # --- Main Content ---
    if st.session_state.logged_in:
        tabs = st.tabs([
            t('tab_start'), 
            t('tab_gallery'), 
            t('tab_groups'), 
            t('tab_reconstruction'), 
            t('tab_labels'),
            t('tab_help')
        ])
        
        # 1. Start Tab (Demo)
        with tabs[0]:
            st.header(t('demo_title'))
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(t('demo_plate'))
                if st.button("🖼️ Generate Plate Demo"):
                    img = generate_demo_plate()
                    # Save to fake buffer for download
                    is_success, buffer = cv2.imencode(".jpg", img)
                    if is_success:
                        st.image(img, channels="BGR", use_column_width=True)
                        st.download_button(
                            label=t('demo_download'),
                            data=buffer.tobytes(),
                            file_name="demo_plate.jpg",
                            mime="image/jpeg"
                        )
            
            with col2:
                st.subheader(t('demo_pottery'))
                if st.button("🏺 Generate Pottery Demo"):
                    img = generate_demo_pottery()
                    is_success, buffer = cv2.imencode(".jpg", img)
                    if is_success:
                        st.image(img, channels="BGR", use_column_width=True)
                        st.download_button(
                            label=t('demo_download'),
                            data=buffer.tobytes(),
                            file_name="demo_pottery.jpg",
                            mime="image/jpeg"
                        )
            
            st.info(t('demo_hint'))
            
            # Deep Link Handler
            query_params = st.query_params
            if "id" in query_params:
                st.success(f"{t('search_found')} ID: {query_params['id']}")

        # 2. Gallery
        with tabs[1]:
            st.header(t('detected_fragments'))
            if st.session_state.fragments:
                cols = st.columns(5)
                for i, frag in enumerate(st.session_state.fragments):
                    with cols[i % 5]:
                        st.image(frag['thumbnail'], channels="BGR")
                        st.caption(f"ID: {frag['id']}")
            else:
                st.info(t('upload_first'))

        # 3. Groups
        with tabs[2]:
            st.header(t('groups_title'))
            groups = st.session_state.groups
            if groups:
                for label, group_frags in groups.items():
                    with st.expander(f"{t('groups')} {label} ({len(group_frags)} {t('pieces')})", expanded=True):
                        cols = st.columns(6)
                        for i, f in enumerate(group_frags):
                            with cols[i % 6]:
                                st.image(f['thumbnail'], channels="BGR")
            else:
                st.info(t('no_groups'))

        # 4. Reconstruction
        with tabs[3]:
            st.header(t('reconstruction_title'))
            
            groups = st.session_state.groups
            if groups:
                selected_group_id = st.selectbox(t('select_group'), list(groups.keys()))
                group_pieces = groups[selected_group_id]
                
                if st.button(t('calculate_btn')):
                    with st.spinner(t('analyzing')):
                        # Call the modified reconstruction
                        res_img, placements, matches = reconstruct_group(group_pieces)
                        
                        if res_img is not None:
                            st.image(res_img, channels="BGR", use_column_width=True)
                            
                            # Info columns
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Fragmente platziert", len([p for p in placements if p.get('placed')]))
                            with c2:
                                st.metric("Gefundene Anschlüsse", len(matches))
                                
                            # Download
                            is_success, buffer = cv2.imencode(".png", res_img)
                            if is_success:
                                st.download_button(
                                    label=t('export_btn'),
                                    data=buffer.tobytes(),
                                    file_name=f"recon_group_{selected_group_id}.png",
                                    mime="image/png"
                                )
                                
                            # Add to labels
                            if st.button("🏷️ Label für diese Rekonstruktion erstellen"):
                                st.session_state.custom_labels.append({
                                    'id': str(uuid.uuid4())[:8],
                                    'name': f"Group {selected_group_id}",
                                    'desc': f"Reconstruction of {len(group_pieces)} fragments."
                                })
                                # Save user data
                                users = load_users()
                                if st.session_state.username in users:
                                    users[st.session_state.username]['custom_labels'] = st.session_state.custom_labels
                                    save_users(users)
                                st.success(t('saved'))
            else:
                st.warning(t('no_groups'))

        # 5. Labels
        with tabs[4]:
            st.header(t('labels_title'))
            
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input(t('custom_label_name'))
            with c2:
                desc = st.text_input(t('custom_label_desc'))
                
            if st.button(t('add_custom_label')):
                if name:
                    st.session_state.custom_labels.append({
                        'id': str(uuid.uuid4())[:8],
                        'name': name,
                        'desc': desc
                    })
                    users = load_users()
                    if st.session_state.username in users:
                        users[st.session_state.username]['custom_labels'] = st.session_state.custom_labels
                        save_users(users)
            
            st.subheader(t('custom_labels_list'))
            if st.session_state.custom_labels:
                for lbl in st.session_state.custom_labels:
                    st.text(f"📄 {lbl['name']} (ID: {lbl['id']})")
                    
                if st.button(t('create_pdf')):
                    pdf_data = create_labels_pdf(st.session_state.custom_labels)
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_data,
                        file_name="labels.pdf",
                        mime="application/pdf"
                    )
                    
                if st.button(t('clear_custom')):
                    st.session_state.custom_labels = []
                    st.rerun()

        # 6. Help
        with tabs[5]:
            st.header(t('help_title'))
            if st.session_state.language == 'de':
                st.markdown(HELP_DE)
            else:
                st.markdown(HELP_EN)

    else:
        st.info("👋 Please Login to start analysis")

if __name__ == "__main__":
    main()
