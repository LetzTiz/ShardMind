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
║   Archäologische Scherben-Analyse & Rekonstruktion                           ║
║   Version 1.2.1                                                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG:
==========

Version 1.2.1 (2025-01-12)
--------------------------
BUGFIXES:
• 🐛 IndexError in reconstruct_group behoben
• 🐛 Datenbank-Anzeige in Sidebar korrigiert
• 📷 QR-Scanner Tab wiederhergestellt
• 🖼️ Beispielbilder-Generator hinzugefügt

Version 1.2 (2025-01-12)
------------------------
• 🔐 LOGIN-SYSTEM mit privater Datenbank
• 🧩 Kanten-basiertes Clustering
• 📖 Hilfe-Tab
• 🌍 Deutsch/English

"""

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
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
import requests
import json
import os
import uuid

# =============================================================================
# KONFIGURATION
# =============================================================================

APP_VERSION = "1.2.1"
USERS_DB_PATH = Path("shardmind_users.pkl")
FEATURE_VERSION = 12

# =============================================================================
# ÜBERSETZUNGEN
# =============================================================================

TRANSLATIONS = {
    'de': {
        'app_title': 'ShardMind',
        'app_subtitle': 'Archäologische Scherben-Analyse & Rekonstruktion',
        'login': 'Anmelden',
        'register': 'Registrieren',
        'username': 'Benutzername',
        'password': 'Passwort',
        'password_confirm': 'Passwort bestätigen',
        'login_btn': '🔐 Anmelden',
        'register_btn': '📝 Registrieren',
        'logout_btn': '🚪 Abmelden',
        'logged_in_as': 'Angemeldet als',
        'login_error': 'Falscher Benutzername oder Passwort',
        'register_success': 'Registrierung erfolgreich! Bitte anmelden.',
        'register_error_exists': 'Benutzername existiert bereits',
        'register_error_password': 'Passwörter stimmen nicht überein',
        'register_error_short': 'Passwort mind. 4 Zeichen',
        'api_settings': '🔑 API',
        'api_key': 'API Key',
        'use_ai': 'KI nutzen',
        'upload_photos': '📤 Fotos hochladen',
        'min_size': 'Min. Größe',
        'cluster_sens': 'Cluster-Sensitivität',
        'edge_weight': 'Kanten-Gewichtung',
        'excavation': 'Grabung',
        'analyze_btn': '🔬 Analysieren',
        'clear_btn': '🗑️ Leeren',
        'language': '🌍 Sprache',
        'tab_gallery': '🏺 Galerie',
        'tab_groups': '📦 Gruppen',
        'tab_reconstruction': '🧩 Rekonstruktion',
        'tab_database': '💾 Datenbank',
        'tab_qr_scanner': '📷 QR-Scanner',
        'tab_help': '❓ Hilfe',
        'tab_examples': '🖼️ Beispiele',
        'detected_fragments': 'Erkannte Fundstücke',
        'groups_title': 'Gruppen',
        'group_name': 'Gruppenname',
        'save_btn': '💾 Speichern',
        'reconstruct_btn': '🧩 Rekonstruieren',
        'pieces': 'Teile',
        'no_groups': 'Keine Gruppen',
        'reconstruction_title': 'Gruppe rekonstruieren',
        'select_group': 'Gruppe wählen',
        'fragments_in_group': 'Fragmente',
        'canvas_size': 'Canvas-Größe',
        'show_edges': 'Kanten zeigen',
        'calculate_btn': '🔄 Rekonstruktion berechnen',
        'analyzing': 'Analysiere...',
        'matches_found': 'Matches gefunden',
        'result_title': 'Ergebnis',
        'export_btn': '📥 Exportieren',
        'edge_matches': 'Kanten-Matches',
        'manual_adjust': 'Manuelle Anpassung',
        'select_fragment': 'Fragment wählen',
        'apply_btn': 'Anwenden',
        'database_title': 'Datenbank',
        'fragments_saved': 'Fundstücke',
        'groups_saved': 'Gruppen',
        'db_empty': 'Datenbank leer',
        'delete_db': '🗑️ Löschen',
        'help_title': 'Hilfe',
        'qr_scanner_title': 'QR-Scanner',
        'qr_upload': 'QR-Code Bild hochladen',
        'qr_result': 'Erkannter Code',
        'qr_found': 'Fundstück gefunden',
        'qr_not_found': 'Nicht in Datenbank',
        'examples_title': 'Beispielbilder generieren',
        'generate_example': 'Beispiel generieren',
        'example_plates': 'Zerbrochene Teller',
        'example_pottery': 'Keramikscherben',
        'fragments': 'Fundstücke',
        'groups': 'Gruppen',
        'upload_first': 'Bitte Fotos hochladen',
        'success_fragments': 'Fundstücke erkannt!',
        'saved': 'Gespeichert!',
    },
    'en': {
        'app_title': 'ShardMind',
        'app_subtitle': 'Archaeological Fragment Analysis & Reconstruction',
        'login': 'Login',
        'register': 'Register',
        'username': 'Username',
        'password': 'Password',
        'password_confirm': 'Confirm Password',
        'login_btn': '🔐 Login',
        'register_btn': '📝 Register',
        'logout_btn': '🚪 Logout',
        'logged_in_as': 'Logged in as',
        'login_error': 'Wrong username or password',
        'register_success': 'Registration successful!',
        'register_error_exists': 'Username already exists',
        'register_error_password': 'Passwords do not match',
        'register_error_short': 'Password min. 4 characters',
        'api_settings': '🔑 API',
        'api_key': 'API Key',
        'use_ai': 'Use AI',
        'upload_photos': '📤 Upload Photos',
        'min_size': 'Min. Size',
        'cluster_sens': 'Cluster Sensitivity',
        'edge_weight': 'Edge Weight',
        'excavation': 'Excavation',
        'analyze_btn': '🔬 Analyze',
        'clear_btn': '🗑️ Clear',
        'language': '🌍 Language',
        'tab_gallery': '🏺 Gallery',
        'tab_groups': '📦 Groups',
        'tab_reconstruction': '🧩 Reconstruction',
        'tab_database': '💾 Database',
        'tab_qr_scanner': '📷 QR Scanner',
        'tab_help': '❓ Help',
        'tab_examples': '🖼️ Examples',
        'detected_fragments': 'Detected Fragments',
        'groups_title': 'Groups',
        'group_name': 'Group Name',
        'save_btn': '💾 Save',
        'reconstruct_btn': '🧩 Reconstruct',
        'pieces': 'pieces',
        'no_groups': 'No groups',
        'reconstruction_title': 'Reconstruct Group',
        'select_group': 'Select Group',
        'fragments_in_group': 'Fragments',
        'canvas_size': 'Canvas Size',
        'show_edges': 'Show edges',
        'calculate_btn': '🔄 Calculate Reconstruction',
        'analyzing': 'Analyzing...',
        'matches_found': 'matches found',
        'result_title': 'Result',
        'export_btn': '📥 Export',
        'edge_matches': 'Edge Matches',
        'manual_adjust': 'Manual Adjustment',
        'select_fragment': 'Select Fragment',
        'apply_btn': 'Apply',
        'database_title': 'Database',
        'fragments_saved': 'Fragments',
        'groups_saved': 'Groups',
        'db_empty': 'Database empty',
        'delete_db': '🗑️ Delete',
        'help_title': 'Help',
        'qr_scanner_title': 'QR Scanner',
        'qr_upload': 'Upload QR Code Image',
        'qr_result': 'Detected Code',
        'qr_found': 'Fragment found',
        'qr_not_found': 'Not in database',
        'examples_title': 'Generate Example Images',
        'generate_example': 'Generate Example',
        'example_plates': 'Broken Plates',
        'example_pottery': 'Pottery Shards',
        'fragments': 'Fragments',
        'groups': 'Groups',
        'upload_first': 'Please upload photos',
        'success_fragments': 'fragments detected!',
        'saved': 'Saved!',
    }
}

def t(key):
    lang = st.session_state.get('language', 'de')
    return TRANSLATIONS.get(lang, TRANSLATIONS['de']).get(key, key)


# =============================================================================
# BEISPIELBILDER GENERIEREN
# =============================================================================

def generate_example_broken_plate(num_pieces=5, plate_color=(180, 120, 80)):
    """Generiert ein Beispielbild mit zerbrochenen Tellerteilen"""
    canvas_size = 800
    img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    
    # Erstelle Tellerform
    center = canvas_size // 2
    radius = 150
    
    # Generiere Bruchlinien
    angles = sorted(np.random.uniform(0, 2*np.pi, num_pieces))
    
    pieces = []
    for i in range(len(angles)):
        angle1 = angles[i]
        angle2 = angles[(i + 1) % len(angles)]
        
        if angle2 < angle1:
            angle2 += 2 * np.pi
        
        # Erstelle Segment
        pts = [(center, center)]
        for a in np.linspace(angle1, angle2, 20):
            x = center + int(radius * np.cos(a))
            y = center + int(radius * np.sin(a))
            pts.append((x, y))
        pts.append((center, center))
        
        pieces.append(np.array(pts, dtype=np.int32))
    
    # Platziere Stücke mit Abstand
    for i, piece_pts in enumerate(pieces):
        # Verschiebe jedes Stück
        offset_x = int(np.random.uniform(-100, 100))
        offset_y = int(np.random.uniform(-100, 100))
        
        shifted = piece_pts.copy()
        shifted[:, 0] += offset_x
        shifted[:, 1] += offset_y
        
        # Füge Variation zur Farbe hinzu
        color_var = tuple(max(0, min(255, c + np.random.randint(-20, 20))) for c in plate_color)
        
        cv2.fillPoly(img, [shifted], color_var)
        cv2.polylines(img, [shifted], True, (100, 70, 50), 2)
    
    return img


def generate_example_pottery_shards(num_shards=6):
    """Generiert ein Beispielbild mit Keramikscherben"""
    canvas_size = 800
    img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 220
    
    colors = [
        (160, 100, 70),   # Terrakotta
        (140, 90, 60),    # Dunkel Terrakotta
        (180, 130, 90),   # Hell Terrakotta
        (120, 80, 50),    # Braun
        (200, 150, 100),  # Beige
    ]
    
    for i in range(num_shards):
        # Zufällige Position
        cx = np.random.randint(100, canvas_size - 100)
        cy = np.random.randint(100, canvas_size - 100)
        
        # Zufällige unregelmäßige Form
        num_pts = np.random.randint(5, 9)
        angles = sorted(np.random.uniform(0, 2*np.pi, num_pts))
        radii = np.random.uniform(30, 70, num_pts)
        
        pts = []
        for angle, r in zip(angles, radii):
            x = cx + int(r * np.cos(angle))
            y = cy + int(r * np.sin(angle))
            pts.append((x, y))
        
        pts = np.array(pts, dtype=np.int32)
        
        color = colors[i % len(colors)]
        color_var = tuple(max(0, min(255, c + np.random.randint(-15, 15))) for c in color)
        
        cv2.fillPoly(img, [pts], color_var)
        cv2.polylines(img, [pts], True, (80, 50, 30), 2)
        
        # Füge Textur hinzu
        for _ in range(5):
            tx = cx + np.random.randint(-30, 30)
            ty = cy + np.random.randint(-30, 30)
            cv2.circle(img, (tx, ty), 1, (color[0]-20, color[1]-20, color[2]-20), -1)
    
    return img


def generate_example_glass_shards(num_shards=7):
    """Generiert ein Beispielbild mit Glasscherben"""
    canvas_size = 800
    img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 200
    
    for i in range(num_shards):
        cx = np.random.randint(100, canvas_size - 100)
        cy = np.random.randint(100, canvas_size - 100)
        
        # Scharfe, kantige Form
        num_pts = np.random.randint(3, 6)
        angles = sorted(np.random.uniform(0, 2*np.pi, num_pts))
        radii = np.random.uniform(20, 60, num_pts)
        
        pts = []
        for angle, r in zip(angles, radii):
            x = cx + int(r * np.cos(angle))
            y = cy + int(r * np.sin(angle))
            pts.append((x, y))
        
        pts = np.array(pts, dtype=np.int32)
        
        # Transparentes Glas-Grün/Blau
        hue = np.random.choice([100, 110, 120, 90])  # Grün-Blau Bereich
        color = (hue + 80, hue + 100, hue + 90)
        
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (150, 180, 170), 1)
    
    return img


# =============================================================================
# BENUTZER-MANAGEMENT
# =============================================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users_db():
    if USERS_DB_PATH.exists():
        try:
            with open(USERS_DB_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {'users': {}}


def save_users_db(users_db):
    with open(USERS_DB_PATH, 'wb') as f:
        pickle.dump(users_db, f)


def register_user(username, password):
    users_db = load_users_db()
    
    if username in users_db['users']:
        return False, 'register_error_exists'
    
    if len(password) < 4:
        return False, 'register_error_short'
    
    users_db['users'][username] = {
        'password_hash': hash_password(password),
        'created': datetime.now().isoformat(),
        'database': {
            'pieces': {},
            'clusters': {},
            'version': FEATURE_VERSION
        }
    }
    
    save_users_db(users_db)
    return True, 'register_success'


def authenticate_user(username, password):
    users_db = load_users_db()
    if username not in users_db['users']:
        return False
    return users_db['users'][username]['password_hash'] == hash_password(password)


def get_user_database(username):
    users_db = load_users_db()
    if username in users_db['users']:
        db = users_db['users'][username].get('database', {})
        if 'pieces' not in db:
            db['pieces'] = {}
        if 'clusters' not in db:
            db['clusters'] = {}
        return db
    return {'pieces': {}, 'clusters': {}, 'version': FEATURE_VERSION}


def save_user_database(username, db):
    users_db = load_users_db()
    if username in users_db['users']:
        users_db['users'][username]['database'] = db
        save_users_db(users_db)


# =============================================================================
# ID-GENERIERUNG
# =============================================================================

def generate_unique_id():
    unique = uuid.uuid4().hex[:8].upper()
    return f"SM-{unique}"


# =============================================================================
# KANTEN-ANALYSE & CLUSTERING
# =============================================================================

def extract_edge_signature(contour, num_points=50):
    """Extrahiert Kanten-Signatur für Matching"""
    contour = contour.squeeze()
    if len(contour.shape) == 1 or len(contour) < 10:
        return None
    
    diffs = np.diff(contour, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total_length = cumulative[-1]
    
    if total_length < 10:
        return None
    
    target_lengths = np.linspace(0, total_length, num_points)
    resampled = np.zeros((num_points, 2))
    
    for i, target in enumerate(target_lengths):
        idx = np.searchsorted(cumulative, target)
        idx = min(idx, len(contour) - 1)
        if idx == 0:
            resampled[i] = contour[0]
        else:
            t_val = (target - cumulative[idx-1]) / (lengths[idx-1] + 1e-6)
            resampled[i] = contour[idx-1] + t_val * (contour[idx] - contour[idx-1])
    
    curvature = np.zeros(num_points)
    for i in range(1, num_points - 1):
        v1 = resampled[i] - resampled[i-1]
        v2 = resampled[i+1] - resampled[i]
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        curvature[i] = np.sin(angle)
    
    return {'points': resampled, 'curvature': curvature, 'total_length': total_length}


def calculate_edge_compatibility(sig1, sig2):
    """Berechnet Kanten-Kompatibilität"""
    if sig1 is None or sig2 is None:
        return 0
    
    c1, c2 = sig1['curvature'], sig2['curvature']
    
    # Normal
    diff = np.mean(np.abs(c1 - c2))
    score1 = max(0, 100 - diff * 100)
    
    # Gespiegelt
    c2_flipped = -c2[::-1]
    diff = np.mean(np.abs(c1 - c2_flipped))
    score2 = max(0, 100 - diff * 100)
    
    best = max(score1, score2)
    
    len_ratio = min(sig1['total_length'], sig2['total_length']) / max(sig1['total_length'], sig2['total_length'])
    
    return best * len_ratio


def cluster_by_reconstruction(pieces, edge_weight=0.6, threshold=40):
    """Clustert basierend auf Rekonstruktions-Potenzial"""
    n = len(pieces)
    if n < 2:
        return [-1] * n
    
    signatures = []
    for p in pieces:
        if 'contour' in p:
            sig = extract_edge_signature(p['contour'])
            signatures.append(sig)
        else:
            signatures.append(None)
    
    compatibility = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            edge_score = calculate_edge_compatibility(signatures[i], signatures[j])
            
            color_score = 0
            if 'features' in pieces[i] and 'features' in pieces[j]:
                color_dist = np.linalg.norm(pieces[i]['features']['color'] - pieces[j]['features']['color'])
                color_score = max(0, 100 - color_dist * 2)
            
            material_bonus = 15 if pieces[i].get('material') == pieces[j].get('material') else 0
            
            combined = edge_weight * edge_score + (1 - edge_weight) * color_score + material_bonus
            compatibility[i, j] = combined
            compatibility[j, i] = combined
    
    distance = 100 - compatibility
    np.fill_diagonal(distance, 0)
    
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=100 - threshold,
            metric='precomputed',
            linkage='average'
        )
        return clustering.fit_predict(distance)
    except:
        return [-1] * n


# =============================================================================
# REKONSTRUKTION (BUGFIX)
# =============================================================================

def find_matching_edges(pieces):
    """Findet passende Kanten"""
    matches = []
    n = len(pieces)
    
    signatures = []
    for p in pieces:
        if 'contour' in p:
            sig = extract_edge_signature(p['contour'])
            signatures.append(sig)
        else:
            signatures.append(None)
    
    for i in range(n):
        for j in range(i + 1, n):
            score = calculate_edge_compatibility(signatures[i], signatures[j])
            if score > 25:
                matches.append({'piece_i': i, 'piece_j': j, 'score': score})
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches


def reconstruct_group(pieces, canvas_size=800):
    """Rekonstruiert Gruppe - MIT BUGFIX"""
    if not pieces:
        return None, [], []
    
    n = len(pieces)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    matches = find_matching_edges(pieces)
    
    # Initialisiere ALLE Placements zuerst
    center = canvas_size // 2
    placements = []
    for i in range(n):
        placements.append({
            'x': center,
            'y': center,
            'rotation': 0,
            'scale': 1.0,
            'placed': False
        })
    
    # Erstes Teil platzieren
    if n > 0:
        placements[0]['placed'] = True
    
    # Platziere basierend auf Matches
    for match in matches:
        i, j = match['piece_i'], match['piece_j']
        
        # Sicherheitscheck
        if i >= n or j >= n:
            continue
        
        if placements[i]['placed'] and not placements[j]['placed']:
            base = placements[i]
            angle = np.random.uniform(0, 2 * np.pi)
            offset = 80 + match['score'] * 0.5
            
            placements[j]['x'] = base['x'] + offset * np.cos(angle)
            placements[j]['y'] = base['y'] + offset * np.sin(angle)
            placements[j]['rotation'] = np.random.uniform(-0.3, 0.3)
            placements[j]['placed'] = True
            placements[j]['match_score'] = match['score']
        
        elif placements[j]['placed'] and not placements[i]['placed']:
            base = placements[j]
            angle = np.random.uniform(0, 2 * np.pi)
            offset = 80 + match['score'] * 0.5
            
            placements[i]['x'] = base['x'] + offset * np.cos(angle)
            placements[i]['y'] = base['y'] + offset * np.sin(angle)
            placements[i]['rotation'] = np.random.uniform(-0.3, 0.3)
            placements[i]['placed'] = True
            placements[i]['match_score'] = match['score']
    
    # Nicht platzierte im Kreis anordnen
    unplaced = [i for i in range(n) if not placements[i]['placed']]
    if unplaced:
        angle_step = 2 * np.pi / len(unplaced)
        radius = canvas_size // 4
        for idx, piece_idx in enumerate(unplaced):
            angle = idx * angle_step
            placements[piece_idx]['x'] = center + radius * np.cos(angle)
            placements[piece_idx]['y'] = center + radius * np.sin(angle)
            placements[piece_idx]['placed'] = True
    
    # Zeichne alle Teile
    for idx in range(n):
        if idx >= len(pieces):
            continue
        
        piece = pieces[idx]
        placement = placements[idx]
        
        if 'thumbnail' not in piece:
            continue
        
        thumb = piece['thumbnail'].copy()
        mask = piece.get('mask', np.ones(thumb.shape[:2], dtype=np.uint8) * 255)
        
        h, w = thumb.shape[:2]
        scale = min(100 / max(h, w), 1.0) * placement.get('scale', 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        try:
            thumb_scaled = cv2.resize(thumb, (new_w, new_h))
            mask_scaled = cv2.resize(mask, (new_w, new_h))
            
            rot_deg = np.degrees(placement.get('rotation', 0))
            rot_center = (new_w // 2, new_h // 2)
            rot_mat = cv2.getRotationMatrix2D(rot_center, rot_deg, 1.0)
            
            cos_v = abs(rot_mat[0, 0])
            sin_v = abs(rot_mat[0, 1])
            new_w_r = int(new_h * sin_v + new_w * cos_v)
            new_h_r = int(new_h * cos_v + new_w * sin_v)
            
            rot_mat[0, 2] += (new_w_r - new_w) / 2
            rot_mat[1, 2] += (new_h_r - new_h) / 2
            
            thumb_rot = cv2.warpAffine(thumb_scaled, rot_mat, (new_w_r, new_h_r), borderValue=(240, 240, 240))
            mask_rot = cv2.warpAffine(mask_scaled, rot_mat, (new_w_r, new_h_r))
            
            x = int(placement['x']) - new_w_r // 2
            y = int(placement['y']) - new_h_r // 2
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(canvas_size, x + new_w_r), min(canvas_size, y + new_h_r)
            
            src_x1, src_y1 = x1 - x, y1 - y
            src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                roi = canvas[y1:y2, x1:x2]
                thumb_roi = thumb_rot[src_y1:src_y2, src_x1:src_x2]
                mask_roi = mask_rot[src_y1:src_y2, src_x1:src_x2]
                
                if roi.shape[:2] == thumb_roi.shape[:2]:
                    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR) / 255.0
                    canvas[y1:y2, x1:x2] = (thumb_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        except Exception as e:
            continue
    
    return canvas, placements, matches


# =============================================================================
# QR-CODE
# =============================================================================

def generate_qr_code(data, size=10):
    qr = qrcode.QRCode(version=1, box_size=size, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def scan_qr_code(image):
    """Scannt QR-Codes aus einem Bild mit OpenCV"""
    # OpenCV QR-Code Detector (keine externe Bibliothek nötig)
    detector = cv2.QRCodeDetector()
    
    results = []
    
    # Versuche QR-Code zu decodieren
    data, vertices, _ = detector.detectAndDecode(image)
    
    if data:
        results.append({
            'data': data,
            'type': 'QRCODE',
            'vertices': vertices
        })
    
    # Versuche auch mit Graustufen
    if not results:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data, vertices, _ = detector.detectAndDecode(gray)
        
        if data:
            results.append({
                'data': data,
                'type': 'QRCODE',
                'vertices': vertices
            })
    
    # Versuche mehrere QR-Codes
    if not results:
        try:
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data:
                        results.append({
                            'data': data,
                            'type': 'QRCODE',
                            'vertices': points[i] if points is not None else None
                        })
        except:
            pass
    
    return results


def parse_shardmind_qr(qr_data):
    """Parst ShardMind QR-Code Daten"""
    if qr_data.startswith('shardmind://find/'):
        parts = qr_data.replace('shardmind://find/', '').split('?')
        piece_id = parts[0]
        excavation = None
        if len(parts) > 1 and 'exc=' in parts[1]:
            excavation = parts[1].split('exc=')[1]
        return {'id': piece_id, 'excavation': excavation}
    return None


# =============================================================================
# BILDVERARBEITUNG
# =============================================================================

def get_dominant_color(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
    if s < 30:
        return "Grau" if v > 80 else "Schwarz"
    elif h < 15 or h > 165:
        return "Rot"
    elif h < 35:
        return "Orange"
    elif h < 75:
        return "Grün"
    elif h < 130:
        return "Blau"
    return "Violett"


def classify_material(piece):
    hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=piece['mask'])
    h, s, v = mean_hsv[:3]
    
    if s < 20 and v > 150:
        return "Glas", "Glasscherbe"
    elif 10 < h < 30 and s > 50:
        return "Keramik", "Gefäßscherbe"
    elif s < 40 and v < 80:
        return "Metall", "Metallfragment"
    return "Unbekannt", "Fragment"


def generate_fragment_name(piece, api_key=None, use_ai=False):
    color = get_dominant_color(piece['thumbnail'], piece['mask'])
    material, obj_type = classify_material(piece)
    
    piece['material'] = material
    piece['object_type'] = obj_type
    piece['color_name'] = color
    
    return f"{material}_{obj_type}_{color}".replace(" ", "_")


def image_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def segment_fragments(image, min_area=100, excavation="", api_key=None, use_ai=False):
    """Segmentiert Fragmente"""
    pad = 30
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    blurred = cv2.GaussianBlur(padded, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh = cv2.bitwise_or(cv2.bitwise_or(thresh1, thresh2), thresh3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > padded.shape[0] * padded.shape[1] * 0.8:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        if w / (h + 1e-6) < 0.1 or w / (h + 1e-6) > 10:
            continue
        
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(padded.shape[1], x + w + margin), min(padded.shape[0], y + h + margin)
        
        roi = padded[y1:y2, x1:x2].copy()
        mask = np.zeros(padded.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask_roi = mask[y1:y2, x1:x2].copy()
        
        contour_rel = c.copy()
        contour_rel[:, :, 0] -= x1
        contour_rel[:, :, 1] -= y1
        
        piece = {
            'id': generate_unique_id(),
            'contour': contour_rel,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': area,
            'deleted': False,
            'excavation': excavation,
            'created': datetime.now().isoformat()
        }
        
        piece['name'] = generate_fragment_name(piece, api_key, use_ai)
        pieces.append(piece)
    
    return pieces


def get_features(piece):
    """Extrahiert Features"""
    M = cv2.moments(piece['contour'])
    if M['m00'] == 0:
        return None
    
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    peri = cv2.arcLength(piece['contour'], True)
    approx = cv2.approxPolyDP(piece['contour'], 0.004 * peri, True)
    pts = approx.squeeze().astype(float)
    
    if pts.ndim == 1 or len(pts) < 3:
        return None
    
    diff = pts - np.array([cx, cy])
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    s_idx = np.argsort(angles)
    
    sig = np.interp(np.linspace(-np.pi, np.pi, 180), angles[s_idx], dists[s_idx], period=2 * np.pi)
    sig = sig / (sig.max() + 1e-6)
    
    lab = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
    lab_mean, lab_std = cv2.meanStdDev(lab, mask=piece['mask'])
    hsv_mean, hsv_std = cv2.meanStdDev(hsv, mask=piece['mask'])
    
    return {
        'shape': sig,
        'color': np.concatenate([lab_mean.flatten(), lab_std.flatten(), hsv_mean.flatten(), hsv_std.flatten()])
    }


def get_cluster_color(cluster_id):
    if cluster_id == -1:
        return "rgb(180, 180, 180)"
    hue = int((cluster_id * 47) % 180)
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


# =============================================================================
# HILFE
# =============================================================================

HELP_DE = """
## 🏺 ShardMind Hilfe

### Was ist ShardMind?
Ein Tool zur Analyse und Rekonstruktion zerbrochener archäologischer Objekte.

### Schnellstart
1. **Fotografieren**: Fragmente auf neutralem Hintergrund
2. **Hochladen**: Bilder in der Sidebar hochladen
3. **Analysieren**: Klick auf "🔬 Analysieren"
4. **Gruppieren**: Automatische Gruppierung ähnlicher Teile
5. **Rekonstruieren**: Im Tab "🧩 Rekonstruktion"

### Tipps
- ✅ Neutraler Hintergrund (weiß/grau)
- ✅ Gute Beleuchtung
- ✅ Teile nicht überlappen
- ✅ Hohe Auflösung

### Parameter
- **Min. Größe**: Kleine Fragmente ignorieren
- **Cluster-Sensitivität**: Höher = strengere Gruppierung
- **Kanten-Gewichtung**: Fokus auf Bruchkanten vs. Farbe
"""

HELP_EN = """
## 🏺 ShardMind Help

### What is ShardMind?
A tool for analyzing and reconstructing broken archaeological objects.

### Quick Start
1. **Photograph**: Fragments on neutral background
2. **Upload**: Upload images in sidebar
3. **Analyze**: Click "🔬 Analyze"
4. **Group**: Automatic grouping of similar pieces
5. **Reconstruct**: In "🧩 Reconstruction" tab

### Tips
- ✅ Neutral background (white/gray)
- ✅ Good lighting
- ✅ Don't overlap pieces
- ✅ High resolution

### Parameters
- **Min. Size**: Ignore small fragments
- **Cluster Sensitivity**: Higher = stricter grouping
- **Edge Weight**: Focus on break edges vs. color
"""


# =============================================================================
# STREAMLIT APP
# =============================================================================

def show_login_page():
    st.title(f"🏺 {t('app_title')}")
    st.caption(t('app_subtitle'))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        lang = st.selectbox("🌍", ['de', 'en'], format_func=lambda x: '🇩🇪' if x == 'de' else '🇬🇧')
        if lang != st.session_state.get('language', 'de'):
            st.session_state.language = lang
            st.rerun()
    
    tab1, tab2 = st.tabs([f"🔐 {t('login')}", f"📝 {t('register')}"])
    
    with tab1:
        with st.form("login"):
            user = st.text_input(t('username'))
            pw = st.text_input(t('password'), type='password')
            if st.form_submit_button(t('login_btn'), use_container_width=True):
                if authenticate_user(user, pw):
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
    
    # Init
    for key, val in [('language', 'de'), ('logged_in', False), ('pieces', []), 
                     ('cluster_names', {}), ('api_key', '')]:
        if key not in st.session_state:
            st.session_state[key] = val
    
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    username = st.session_state.username
    db = get_user_database(username)
    
    # SIDEBAR
    with st.sidebar:
        st.title(f"🏺 {t('app_title')}")
        st.caption(f"v{APP_VERSION}")
        
        st.success(f"👤 {username}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t('logout_btn'), use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.pieces = []
                st.rerun()
        with col2:
            lang = st.selectbox("", ['de', 'en'], format_func=lambda x: '🇩🇪' if x == 'de' else '🇬🇧',
                               index=0 if st.session_state.language == 'de' else 1, key='lang_main')
            if lang != st.session_state.language:
                st.session_state.language = lang
                st.rerun()
        
        st.markdown("---")
        
        with st.expander(t('api_settings'), expanded=False):
            api_key = st.text_input(t('api_key'), value=st.session_state.api_key, type='password')
            st.session_state.api_key = api_key
            use_ai = st.checkbox(t('use_ai'), value=bool(api_key), disabled=not api_key)
        
        st.markdown("---")
        
        files = st.file_uploader(t('upload_photos'), type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        min_area = st.slider(t('min_size'), 50, 1000, 200)
        cluster_threshold = st.slider(t('cluster_sens'), 10, 90, 40)
        edge_weight = st.slider(t('edge_weight'), 0.0, 1.0, 0.6, 0.1)
        
        excavation = st.text_input(t('excavation'), value=f"Excavation_{datetime.now().strftime('%Y')}")
        
        st.markdown("---")
        
        if st.button(t('analyze_btn'), type='primary', use_container_width=True):
            if files:
                with st.spinner(t('analyzing')):
                    all_pieces = []
                    progress = st.progress(0)
                    
                    for i, f in enumerate(files):
                        img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        pieces = segment_fragments(img, min_area, excavation, api_key if use_ai else None, use_ai)
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
            st.rerun()
        
        st.markdown("---")
        
        # KORRIGIERTE Datenbank-Anzeige
        db_pieces = db.get('pieces', {})
        db_clusters = db.get('clusters', {})
        st.metric(t('fragments'), len(db_pieces) if isinstance(db_pieces, dict) else 0)
        st.metric(t('groups'), len(db_clusters) if isinstance(db_clusters, dict) else 0)
    
    # HAUPTBEREICH
    if st.session_state.pieces:
        active = [p for p in st.session_state.pieces if not p.get('deleted')]
        
        if len(active) > 1:
            labels = cluster_by_reconstruction(active, edge_weight, cluster_threshold)
            for i, p in enumerate(active):
                p['cluster'] = labels[i]
        else:
            for p in active:
                p['cluster'] = -1
        
        cluster_ids = set([p.get('cluster', -1) for p in active])
        n_clusters = len([c for c in cluster_ids if c >= 0])
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"🏺 {t('fragments')}", len(active))
        col2.metric(f"📦 {t('groups')}", n_clusters)
        col3.metric(f"🗺️", excavation[:15])
        
        # TABS
        tabs = st.tabs([
            t('tab_gallery'),
            t('tab_groups'),
            t('tab_reconstruction'),
            t('tab_database'),
            t('tab_qr_scanner'),
            t('tab_examples'),
            t('tab_help')
        ])
        
        # GALERIE
        with tabs[0]:
            st.header(t('detected_fragments'))
            cols = st.columns(5)
            for i, p in enumerate(active):
                with cols[i % 5]:
                    color = get_cluster_color(p.get('cluster', -1))
                    st.markdown(
                        f'<div style="border:3px solid {color};padding:3px;border-radius:5px;">'
                        f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;"></div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"**{p['id'][:11]}**\n{p.get('name', '')[:18]}")
        
        # GRUPPEN
        with tabs[1]:
            st.header(t('groups_title'))
            
            cluster_sorted = sorted([c for c in cluster_ids if c >= 0])
            
            if not cluster_sorted:
                st.warning(t('no_groups'))
            else:
                for cid in cluster_sorted:
                    cluster_pieces = [p for p in active if p.get('cluster') == cid]
                    materials = [p.get('material', 'Unknown') for p in cluster_pieces]
                    common = max(set(materials), key=materials.count) if materials else 'Unknown'
                    default_name = st.session_state.cluster_names.get(cid, f"{common}_Group_{cid}")
                    
                    with st.expander(f"📦 {default_name} ({len(cluster_pieces)} {t('pieces')})", expanded=True):
                        name = st.text_input(t('group_name'), value=default_name, key=f"n_{cid}")
                        st.session_state.cluster_names[cid] = name
                        
                        pcols = st.columns(min(6, len(cluster_pieces)))
                        for i, p in enumerate(cluster_pieces[:6]):
                            with pcols[i]:
                                st.image(p['thumbnail'])
                                st.caption(p['id'][:8])
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button(t('save_btn'), key=f"s_{cid}"):
                                key = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                db['clusters'][key] = {
                                    'name': name,
                                    'created': datetime.now().isoformat(),
                                    'piece_ids': [p['id'] for p in cluster_pieces]
                                }
                                for p in cluster_pieces:
                                    db['pieces'][p['id']] = p.copy()
                                save_user_database(username, db)
                                st.success(t('saved'))
        
        # REKONSTRUKTION
        with tabs[2]:
            st.header(t('reconstruction_title'))
            
            cluster_opts = {
                f"{st.session_state.cluster_names.get(c, f'G{c}')} ({sum(1 for p in active if p.get('cluster')==c)})": c
                for c in cluster_ids if c >= 0
            }
            
            if cluster_opts:
                sel_name = st.selectbox(t('select_group'), list(cluster_opts.keys()))
                sel_id = cluster_opts[sel_name]
                
                recon_pieces = [p for p in active if p.get('cluster') == sel_id]
                st.write(f"**{len(recon_pieces)} {t('fragments_in_group')}**")
                
                pcols = st.columns(min(8, len(recon_pieces)))
                for i, p in enumerate(recon_pieces[:8]):
                    with pcols[i]:
                        st.image(p['thumbnail'], use_container_width=True)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    canvas_size = st.slider(t('canvas_size'), 400, 1200, 800, 100)
                with c2:
                    show_edges = st.checkbox(t('show_edges'), value=True)
                
                if st.button(t('calculate_btn'), type='primary', use_container_width=True):
                    with st.spinner(t('analyzing')):
                        img, plc, mtch = reconstruct_group(recon_pieces, canvas_size)
                        if img is not None:
                            st.session_state.recon_image = img
                            st.session_state.recon_placements = plc
                            st.session_state.recon_matches = mtch
                            st.success(f"✓ {len(mtch)} {t('matches_found')}")
                
                if 'recon_image' in st.session_state and st.session_state.recon_image is not None:
                    st.markdown(f"### {t('result_title')}")
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        rgb = cv2.cvtColor(st.session_state.recon_image, cv2.COLOR_BGR2RGB)
                        st.image(rgb, use_container_width=True)
                        
                        buf = io.BytesIO()
                        Image.fromarray(rgb).save(buf, format='PNG')
                        st.download_button(t('export_btn'), buf.getvalue(), f"recon_{sel_id}.png", "image/png")
                    
                    with c2:
                        st.markdown(f"**{t('edge_matches')}**")
                        for i, m in enumerate(st.session_state.get('recon_matches', [])[:5]):
                            st.write(f"Match {i+1}: {m['score']:.0f}%")
                        
                        st.markdown("---")
                        st.markdown(f"**{t('manual_adjust')}**")
                        
                        if recon_pieces and st.session_state.get('recon_placements'):
                            pidx = st.selectbox(t('select_fragment'), range(len(recon_pieces)),
                                               format_func=lambda x: recon_pieces[x]['id'][:11] if x < len(recon_pieces) else '')
                            
                            if pidx < len(st.session_state.recon_placements):
                                pl = st.session_state.recon_placements[pidx]
                                nx = st.slider("X", 0, canvas_size, int(pl.get('x', 400)))
                                ny = st.slider("Y", 0, canvas_size, int(pl.get('y', 400)))
                                nr = st.slider("°", -180, 180, int(np.degrees(pl.get('rotation', 0))))
                                ns = st.slider("Scale", 0.5, 2.0, pl.get('scale', 1.0), 0.1)
                                
                                if st.button(t('apply_btn')):
                                    st.session_state.recon_placements[pidx] = {
                                        'x': nx, 'y': ny, 'rotation': np.radians(nr), 'scale': ns, 'placed': True
                                    }
                                    img, _, _ = reconstruct_group(recon_pieces, canvas_size)
                                    st.session_state.recon_image = img
                                    st.rerun()
                
                if show_edges and recon_pieces:
                    st.markdown("### Edges")
                    ecols = st.columns(min(4, len(recon_pieces)))
                    for i, p in enumerate(recon_pieces[:4]):
                        with ecols[i]:
                            vis = p['thumbnail'].copy()
                            if 'contour' in p:
                                cv2.drawContours(vis, [p['contour']], -1, (0, 255, 0), 2)
                            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            else:
                st.warning(t('no_groups'))
        
        # DATENBANK
        with tabs[3]:
            st.header(t('database_title'))
            
            db_pieces = db.get('pieces', {})
            if db_pieces:
                st.write(f"**{len(db_pieces)} {t('fragments_saved')}**")
                
                dcols = st.columns(6)
                for i, (pid, piece) in enumerate(list(db_pieces.items())[:12]):
                    with dcols[i % 6]:
                        if 'thumbnail' in piece:
                            st.image(piece['thumbnail'])
                        st.caption(pid[:11])
            else:
                st.info(t('db_empty'))
            
            if db.get('clusters'):
                st.markdown("---")
                st.write(f"**{t('groups_saved')}**")
                for k, v in db['clusters'].items():
                    st.write(f"• {v['name']} ({len(v.get('piece_ids', []))} {t('pieces')})")
        
        # QR-SCANNER
        with tabs[4]:
            st.header(t('qr_scanner_title'))
            
            qr_file = st.file_uploader(t('qr_upload'), type=['png', 'jpg', 'jpeg'], key='qr_upload')
            
            if qr_file:
                img = cv2.imdecode(np.asarray(bytearray(qr_file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)
                
                results = scan_qr_code(img)
                
                if results:
                    for r in results:
                        st.success(f"**{t('qr_result')}:** `{r['data']}`")
                        
                        parsed = parse_shardmind_qr(r['data'])
                        if parsed:
                            piece_id = parsed['id']
                            
                            # Suche in aktueller Session
                            found = None
                            for p in active:
                                if p['id'] == piece_id:
                                    found = p
                                    break
                            
                            # Suche in Datenbank
                            if not found and piece_id in db.get('pieces', {}):
                                found = db['pieces'][piece_id]
                            
                            if found:
                                st.success(f"✓ {t('qr_found')}")
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    if 'thumbnail' in found:
                                        st.image(found['thumbnail'])
                                with c2:
                                    st.write(f"**ID:** {found['id']}")
                                    st.write(f"**Name:** {found.get('name', 'N/A')}")
                                    st.write(f"**Material:** {found.get('material', 'N/A')}")
                                    st.write(f"**Grabung:** {found.get('excavation', 'N/A')}")
                            else:
                                st.warning(f"⚠️ {t('qr_not_found')}: {piece_id}")
                else:
                    st.warning("Kein QR-Code gefunden / No QR code found")
        
        # BEISPIELE
        with tabs[5]:
            st.header(t('examples_title'))
            
            st.info("Generiere Testbilder um ShardMind auszuprobieren / Generate test images to try ShardMind")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader(f"🍽️ {t('example_plates')}")
                num_plate = st.slider("Teile/Pieces", 3, 8, 5, key='plate_n')
                if st.button(t('generate_example'), key='gen_plate'):
                    img = generate_example_broken_plate(num_plate)
                    st.session_state.example_plate = img
                
                if 'example_plate' in st.session_state:
                    st.image(cv2.cvtColor(st.session_state.example_plate, cv2.COLOR_BGR2RGB))
                    
                    buf = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(st.session_state.example_plate, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
                    st.download_button("📥 Download", buf.getvalue(), "broken_plate.png", "image/png")
            
            with c2:
                st.subheader(f"🏺 {t('example_pottery')}")
                num_pot = st.slider("Teile/Pieces", 3, 10, 6, key='pot_n')
                if st.button(t('generate_example'), key='gen_pot'):
                    img = generate_example_pottery_shards(num_pot)
                    st.session_state.example_pottery = img
                
                if 'example_pottery' in st.session_state:
                    st.image(cv2.cvtColor(st.session_state.example_pottery, cv2.COLOR_BGR2RGB))
                    
                    buf = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(st.session_state.example_pottery, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
                    st.download_button("📥 Download", buf.getvalue(), "pottery.png", "image/png")
            
            with c3:
                st.subheader("🔮 Glas / Glass")
                num_glass = st.slider("Teile/Pieces", 3, 10, 7, key='glass_n')
                if st.button(t('generate_example'), key='gen_glass'):
                    img = generate_example_glass_shards(num_glass)
                    st.session_state.example_glass = img
                
                if 'example_glass' in st.session_state:
                    st.image(cv2.cvtColor(st.session_state.example_glass, cv2.COLOR_BGR2RGB))
                    
                    buf = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(st.session_state.example_glass, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
                    st.download_button("📥 Download", buf.getvalue(), "glass.png", "image/png")
        
        # HILFE
        with tabs[6]:
            st.header(t('help_title'))
            st.markdown(HELP_DE if st.session_state.language == 'de' else HELP_EN)
    
    else:
        # Startseite
        st.title(f"🏺 {t('app_title')} v{APP_VERSION}")
        st.markdown(f"### {t('app_subtitle')}")
        
        st.markdown(HELP_DE if st.session_state.language == 'de' else HELP_EN)
        
        st.markdown("---")
        st.info("👈 Lade Fotos in der Sidebar hoch um zu starten / Upload photos in the sidebar to start")


if __name__ == "__main__":
    main()
