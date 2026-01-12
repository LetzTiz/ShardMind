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
║   Version 1.2                                                                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG:
==========

Version 1.2 (2025-01-12)
------------------------
NEUE FEATURES:
• 🔐 LOGIN-SYSTEM: Benutzer mit eigener privater Datenbank
• 🧩 VERBESSERTES CLUSTERING: Gruppiert nur rekonstruierbare Teile
• 📖 HILFE-TAB: Ausführliche Anleitung und Tutorial
• 🌍 ZWEISPRACHIG: Deutsch / English umschaltbar
• Kanten-basiertes Clustering für bessere Rekonstruktion

Version 1.1 (2025-01-12)
------------------------
• Rekonstruktions-Modus für Gruppen
• Automatische Kanten-Analyse und Matching
• Export der Rekonstruktion als Bild

Version 1.0 (2025-01-11)
------------------------
• Eindeutige UUIDs für alle Fundstücke
• KI-gestützte Objekterkennung
• PDF-Export mit QR-Codes

"""

import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
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

APP_VERSION = "1.2"
USERS_DB_PATH = Path("shardmind_users.pkl")
FEATURE_VERSION = 12

# =============================================================================
# ÜBERSETZUNGEN
# =============================================================================

TRANSLATIONS = {
    'de': {
        # App
        'app_title': 'ShardMind',
        'app_subtitle': 'Archäologische Scherben-Analyse & Rekonstruktion',
        'version': 'Version',
        
        # Login
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
        'register_error_short': 'Passwort muss mindestens 4 Zeichen haben',
        
        # Sidebar
        'api_settings': '🔑 API Einstellungen',
        'api_key': 'API Key',
        'use_ai': 'KI nutzen',
        'upload_photos': '📤 Fotos hochladen',
        'parameters': '⚙️ Parameter',
        'min_size': 'Min. Größe',
        'cluster_sens': 'Cluster-Sensitivität',
        'edge_weight': 'Kanten-Gewichtung',
        'excavation': 'Grabung',
        'analyze_btn': '🔬 Analysieren',
        'clear_btn': '🗑️ Leeren',
        'language': '🌍 Sprache',
        
        # Tabs
        'tab_gallery': '🏺 Galerie',
        'tab_groups': '📦 Gruppen',
        'tab_reconstruction': '🧩 Rekonstruktion',
        'tab_database': '💾 Datenbank',
        'tab_help': '❓ Hilfe',
        
        # Gallery
        'detected_fragments': 'Erkannte Fundstücke',
        'recluster_btn': '🔄 Neu gruppieren',
        
        # Groups
        'groups_title': 'Gruppen',
        'group_name': 'Gruppenname',
        'save_btn': '💾 Speichern',
        'reconstruct_btn': '🧩 Rekonstruieren',
        'pieces': 'Teile',
        'no_groups': 'Keine Gruppen vorhanden',
        'increase_sensitivity': 'Erhöhe die Cluster-Sensitivität',
        
        # Reconstruction
        'reconstruction_title': 'Gruppe rekonstruieren',
        'reconstruction_info': '''**So funktioniert's:**
1. Wähle eine Gruppe aus
2. Klicke "Rekonstruktion berechnen"
3. ShardMind analysiert die Bruchkanten
4. Die Fragmente werden automatisch zusammengesetzt
5. Passe bei Bedarf manuell an''',
        'select_group': 'Gruppe wählen',
        'fragments_in_group': 'Fragmente in dieser Gruppe',
        'canvas_size': 'Canvas-Größe',
        'auto_arrange': 'Auto-Anordnung',
        'show_edges': 'Kanten zeigen',
        'calculate_btn': '🔄 Rekonstruktion berechnen',
        'analyzing': 'Analysiere Bruchkanten...',
        'matches_found': 'Kanten-Matches gefunden',
        'result_title': 'Rekonstruktions-Ergebnis',
        'export_btn': '📥 Als Bild exportieren',
        'edge_matches': 'Kanten-Matches',
        'match': 'Match',
        'manual_adjust': 'Manuelle Anpassung',
        'select_fragment': 'Fragment wählen',
        'x_position': 'X-Position',
        'y_position': 'Y-Position',
        'rotation': 'Rotation',
        'scale': 'Skalierung',
        'apply_btn': 'Anwenden',
        'edge_analysis': 'Kanten-Analyse',
        'edges': 'Kanten',
        
        # Database
        'database_title': 'Datenbank',
        'fragments_saved': 'Fundstücke gespeichert',
        'groups_saved': 'Gruppen',
        'db_empty': 'Datenbank ist leer',
        'delete_db': '🗑️ Datenbank löschen',
        'delete_confirm': 'Zum Löschen "LÖSCHEN" eingeben',
        'delete_btn': '🔥 Endgültig löschen',
        
        # Help
        'help_title': 'Hilfe & Anleitung',
        'help_overview': 'Übersicht',
        'help_workflow': 'Workflow',
        'help_tips': 'Tipps',
        'help_faq': 'FAQ',
        
        # Messages
        'upload_first': 'Bitte Fotos hochladen',
        'analyzing_msg': 'Analysiere...',
        'success_fragments': 'Fundstücke erkannt!',
        'saved': 'Gespeichert!',
        'no_groups_warning': 'Keine Gruppen vorhanden. Analysiere zuerst Bilder.',
        
        # Metrics
        'fragments': 'Fundstücke',
        'groups': 'Gruppen',
    },
    
    'en': {
        # App
        'app_title': 'ShardMind',
        'app_subtitle': 'Archaeological Fragment Analysis & Reconstruction',
        'version': 'Version',
        
        # Login
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
        'register_success': 'Registration successful! Please login.',
        'register_error_exists': 'Username already exists',
        'register_error_password': 'Passwords do not match',
        'register_error_short': 'Password must be at least 4 characters',
        
        # Sidebar
        'api_settings': '🔑 API Settings',
        'api_key': 'API Key',
        'use_ai': 'Use AI',
        'upload_photos': '📤 Upload Photos',
        'parameters': '⚙️ Parameters',
        'min_size': 'Min. Size',
        'cluster_sens': 'Cluster Sensitivity',
        'edge_weight': 'Edge Weight',
        'excavation': 'Excavation',
        'analyze_btn': '🔬 Analyze',
        'clear_btn': '🗑️ Clear',
        'language': '🌍 Language',
        
        # Tabs
        'tab_gallery': '🏺 Gallery',
        'tab_groups': '📦 Groups',
        'tab_reconstruction': '🧩 Reconstruction',
        'tab_database': '💾 Database',
        'tab_help': '❓ Help',
        
        # Gallery
        'detected_fragments': 'Detected Fragments',
        'recluster_btn': '🔄 Re-cluster',
        
        # Groups
        'groups_title': 'Groups',
        'group_name': 'Group Name',
        'save_btn': '💾 Save',
        'reconstruct_btn': '🧩 Reconstruct',
        'pieces': 'pieces',
        'no_groups': 'No groups available',
        'increase_sensitivity': 'Increase cluster sensitivity',
        
        # Reconstruction
        'reconstruction_title': 'Reconstruct Group',
        'reconstruction_info': '''**How it works:**
1. Select a group
2. Click "Calculate Reconstruction"
3. ShardMind analyzes the break edges
4. Fragments are automatically assembled
5. Adjust manually if needed''',
        'select_group': 'Select Group',
        'fragments_in_group': 'Fragments in this group',
        'canvas_size': 'Canvas Size',
        'auto_arrange': 'Auto-arrange',
        'show_edges': 'Show edges',
        'calculate_btn': '🔄 Calculate Reconstruction',
        'analyzing': 'Analyzing break edges...',
        'matches_found': 'edge matches found',
        'result_title': 'Reconstruction Result',
        'export_btn': '📥 Export as Image',
        'edge_matches': 'Edge Matches',
        'match': 'Match',
        'manual_adjust': 'Manual Adjustment',
        'select_fragment': 'Select Fragment',
        'x_position': 'X Position',
        'y_position': 'Y Position',
        'rotation': 'Rotation',
        'scale': 'Scale',
        'apply_btn': 'Apply',
        'edge_analysis': 'Edge Analysis',
        'edges': 'Edges',
        
        # Database
        'database_title': 'Database',
        'fragments_saved': 'fragments saved',
        'groups_saved': 'Groups',
        'db_empty': 'Database is empty',
        'delete_db': '🗑️ Delete Database',
        'delete_confirm': 'Type "DELETE" to confirm',
        'delete_btn': '🔥 Delete permanently',
        
        # Help
        'help_title': 'Help & Guide',
        'help_overview': 'Overview',
        'help_workflow': 'Workflow',
        'help_tips': 'Tips',
        'help_faq': 'FAQ',
        
        # Messages
        'upload_first': 'Please upload photos',
        'analyzing_msg': 'Analyzing...',
        'success_fragments': 'fragments detected!',
        'saved': 'Saved!',
        'no_groups_warning': 'No groups available. Analyze images first.',
        
        # Metrics
        'fragments': 'Fragments',
        'groups': 'Groups',
    }
}

def t(key):
    """Übersetzungsfunktion"""
    lang = st.session_state.get('language', 'de')
    return TRANSLATIONS.get(lang, TRANSLATIONS['de']).get(key, key)


# =============================================================================
# BENUTZER-MANAGEMENT
# =============================================================================

def hash_password(password):
    """Hasht ein Passwort mit SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users_db():
    """Lädt die Benutzer-Datenbank"""
    if USERS_DB_PATH.exists():
        try:
            with open(USERS_DB_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {'users': {}}


def save_users_db(users_db):
    """Speichert die Benutzer-Datenbank"""
    with open(USERS_DB_PATH, 'wb') as f:
        pickle.dump(users_db, f)


def register_user(username, password):
    """Registriert einen neuen Benutzer"""
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
            'reconstructions': {},
            'version': FEATURE_VERSION
        }
    }
    
    save_users_db(users_db)
    return True, 'register_success'


def authenticate_user(username, password):
    """Authentifiziert einen Benutzer"""
    users_db = load_users_db()
    
    if username not in users_db['users']:
        return False
    
    return users_db['users'][username]['password_hash'] == hash_password(password)


def get_user_database(username):
    """Holt die Datenbank eines Benutzers"""
    users_db = load_users_db()
    
    if username in users_db['users']:
        return users_db['users'][username].get('database', {
            'pieces': {},
            'clusters': {},
            'reconstructions': {},
            'version': FEATURE_VERSION
        })
    
    return {'pieces': {}, 'clusters': {}, 'reconstructions': {}, 'version': FEATURE_VERSION}


def save_user_database(username, db):
    """Speichert die Datenbank eines Benutzers"""
    users_db = load_users_db()
    
    if username in users_db['users']:
        users_db['users'][username]['database'] = db
        save_users_db(users_db)


# =============================================================================
# ID-GENERIERUNG
# =============================================================================

def generate_unique_id():
    """Generiert eine eindeutige ID"""
    unique = uuid.uuid4().hex[:8].upper()
    return f"SM-{unique}"


# =============================================================================
# KANTEN-BASIERTES CLUSTERING
# =============================================================================

def extract_edge_signature(contour, num_points=50):
    """
    Extrahiert eine Kanten-Signatur für das Matching.
    Diese Signatur beschreibt die Form der Bruchkante.
    """
    contour = contour.squeeze()
    if len(contour.shape) == 1 or len(contour) < 10:
        return None
    
    # Berechne Gesamtlänge
    diffs = np.diff(contour, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total_length = cumulative[-1]
    
    if total_length < 10:
        return None
    
    # Resample
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
    
    # Berechne Krümmung
    curvature = np.zeros(num_points)
    for i in range(1, num_points - 1):
        v1 = resampled[i] - resampled[i-1]
        v2 = resampled[i+1] - resampled[i]
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        curvature[i] = np.sin(angle)  # Normalisiert auf [-1, 1]
    
    return {
        'points': resampled,
        'curvature': curvature,
        'total_length': total_length
    }


def calculate_edge_compatibility(sig1, sig2):
    """
    Berechnet wie gut zwei Kanten zusammenpassen könnten.
    Hoher Wert = könnten vom gleichen zerbrochenen Objekt stammen.
    """
    if sig1 is None or sig2 is None:
        return 0
    
    # Vergleiche Krümmungsprofile
    c1 = sig1['curvature']
    c2 = sig2['curvature']
    
    # Teste verschiedene Ausrichtungen (gespiegelt, rotiert)
    best_score = 0
    
    # Normal
    diff = np.mean(np.abs(c1 - c2))
    score = max(0, 100 - diff * 100)
    best_score = max(best_score, score)
    
    # Gespiegelt (Bruchkanten passen gespiegelt zusammen)
    c2_flipped = -c2[::-1]
    diff = np.mean(np.abs(c1 - c2_flipped))
    score = max(0, 100 - diff * 100)
    best_score = max(best_score, score)
    
    # Längenähnlichkeit
    len_ratio = min(sig1['total_length'], sig2['total_length']) / max(sig1['total_length'], sig2['total_length'])
    
    return best_score * len_ratio


def cluster_by_reconstruction_potential(pieces, edge_weight=0.6, color_weight=0.4, threshold=40):
    """
    Clustert Teile basierend auf Rekonstruktions-Potenzial.
    Kombiniert Kanten-Kompatibilität und Farb-Ähnlichkeit.
    """
    n = len(pieces)
    if n < 2:
        return [-1] * n
    
    # Extrahiere Kanten-Signaturen
    signatures = []
    for p in pieces:
        if 'contour' in p:
            sig = extract_edge_signature(p['contour'])
            signatures.append(sig)
        else:
            signatures.append(None)
    
    # Berechne Kompatibilitäts-Matrix
    compatibility_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Kanten-Kompatibilität
            edge_score = calculate_edge_compatibility(signatures[i], signatures[j])
            
            # Farb-Ähnlichkeit
            color_score = 0
            if 'features' in pieces[i] and 'features' in pieces[j]:
                color_dist = np.linalg.norm(
                    pieces[i]['features']['color'] - pieces[j]['features']['color']
                )
                color_score = max(0, 100 - color_dist * 2)
            
            # Material-Bonus (gleiches Material = wahrscheinlicher zusammengehörig)
            material_bonus = 20 if pieces[i].get('material') == pieces[j].get('material') else 0
            
            # Kombinierter Score
            combined_score = (edge_weight * edge_score + 
                            color_weight * color_score + 
                            material_bonus)
            
            compatibility_matrix[i, j] = combined_score
            compatibility_matrix[j, i] = combined_score
    
    # Clustering basierend auf Kompatibilität
    # Konvertiere zu Distanz-Matrix
    distance_matrix = 100 - compatibility_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Agglomeratives Clustering
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=100 - threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
    except:
        # Fallback
        labels = [-1] * n
    
    return labels


# =============================================================================
# REKONSTRUKTIONS-ALGORITHMUS
# =============================================================================

def find_matching_edges(pieces):
    """Findet passende Kanten zwischen Fragmenten"""
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
            if score > 30:
                matches.append({
                    'piece_i': i,
                    'piece_j': j,
                    'score': score
                })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches


def reconstruct_group(pieces, canvas_size=800):
    """Rekonstruiert eine Gruppe von Fragmenten"""
    if not pieces:
        return None, [], []
    
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    matches = find_matching_edges(pieces)
    
    # Platzierungen initialisieren
    placements = []
    placed = [False] * len(pieces)
    
    # Erstes Teil in der Mitte
    center = canvas_size // 2
    placements.append({'x': center, 'y': center, 'rotation': 0, 'scale': 1.0})
    placed[0] = True
    
    # Platziere restliche Teile
    for match in matches:
        i, j = match['piece_i'], match['piece_j']
        
        if placed[i] and not placed[j]:
            base = placements[i]
            offset = 100 + match['score'] * 0.5
            angle = len([p for p in placed if p]) * 0.8
            
            new_x = base['x'] + offset * np.cos(angle)
            new_y = base['y'] + offset * np.sin(angle)
            
            placements.append({
                'x': new_x, 'y': new_y,
                'rotation': angle * 0.3,
                'scale': 1.0,
                'match_score': match['score']
            })
            placed[j] = True
        
        elif placed[j] and not placed[i]:
            base = placements[j] if j < len(placements) else placements[0]
            offset = 100 + match['score'] * 0.5
            angle = len([p for p in placed if p]) * 0.8
            
            new_x = base['x'] + offset * np.cos(angle)
            new_y = base['y'] + offset * np.sin(angle)
            
            while len(placements) <= i:
                placements.append({'x': center, 'y': center, 'rotation': 0, 'scale': 1.0})
            
            placements[i] = {
                'x': new_x, 'y': new_y,
                'rotation': angle * 0.3,
                'scale': 1.0,
                'match_score': match['score']
            }
            placed[i] = True
    
    # Nicht platzierte Teile im Kreis anordnen
    angle_step = 2 * np.pi / max(sum(1 for p in placed if not p), 1)
    current_angle = 0
    radius = canvas_size // 4
    
    for idx, is_placed in enumerate(placed):
        if not is_placed:
            x = center + int(radius * np.cos(current_angle))
            y = center + int(radius * np.sin(current_angle))
            while len(placements) <= idx:
                placements.append({'x': center, 'y': center, 'rotation': 0, 'scale': 1.0})
            placements[idx] = {'x': x, 'y': y, 'rotation': 0, 'scale': 1.0}
            current_angle += angle_step
    
    # Zeichne Teile
    for idx, piece in enumerate(pieces):
        if idx >= len(placements) or 'thumbnail' not in piece:
            continue
        
        placement = placements[idx]
        thumb = piece['thumbnail'].copy()
        mask = piece.get('mask', np.ones(thumb.shape[:2], dtype=np.uint8) * 255)
        
        h, w = thumb.shape[:2]
        scale = min(120 / max(h, w), 1.0) * placement.get('scale', 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        thumb_scaled = cv2.resize(thumb, (new_w, new_h))
        mask_scaled = cv2.resize(mask, (new_w, new_h))
        
        # Rotation
        rotation_deg = np.degrees(placement.get('rotation', 0))
        rot_center = (new_w // 2, new_h // 2)
        rot_matrix = cv2.getRotationMatrix2D(rot_center, rotation_deg, 1.0)
        
        cos_val = abs(rot_matrix[0, 0])
        sin_val = abs(rot_matrix[0, 1])
        new_w_rot = int(new_h * sin_val + new_w * cos_val)
        new_h_rot = int(new_h * cos_val + new_w * sin_val)
        
        rot_matrix[0, 2] += (new_w_rot - new_w) / 2
        rot_matrix[1, 2] += (new_h_rot - new_h) / 2
        
        thumb_rot = cv2.warpAffine(thumb_scaled, rot_matrix, (new_w_rot, new_h_rot), borderValue=(240, 240, 240))
        mask_rot = cv2.warpAffine(mask_scaled, rot_matrix, (new_w_rot, new_h_rot))
        
        # Position
        x = int(placement['x']) - new_w_rot // 2
        y = int(placement['y']) - new_h_rot // 2
        
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(canvas_size, x + new_w_rot), min(canvas_size, y + new_h_rot)
        
        src_x1, src_y1 = x1 - x, y1 - y
        src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1 and src_x2 > src_x1 and src_y2 > src_y1:
            try:
                roi = canvas[y1:y2, x1:x2]
                thumb_roi = thumb_rot[src_y1:src_y2, src_x1:src_x2]
                mask_roi = mask_rot[src_y1:src_y2, src_x1:src_x2]
                
                if roi.shape[:2] == thumb_roi.shape[:2]:
                    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR) / 255.0
                    canvas[y1:y2, x1:x2] = (thumb_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
            except:
                pass
    
    return canvas, placements, matches


# =============================================================================
# KI & BILDVERARBEITUNG
# =============================================================================

def analyze_fragment_with_ai(image_base64, api_key=None):
    """KI-basierte Fragment-Analyse"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {"type": "text", "text": "Analyze this fragment. Reply ONLY with JSON: {\"material\": \"type\", \"object_type\": \"type\", \"color\": \"color\"}"}
                    ]
                }]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            text = response.json()['content'][0]['text'].strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
    except:
        pass
    return None


def get_dominant_color(image, mask):
    """Ermittelt die dominante Farbe"""
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
    else:
        return "Violett"


def classify_material(piece):
    """Heuristische Material-Klassifikation"""
    hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=piece['mask'])
    h, s, v = mean_hsv[:3]
    
    if s < 20 and v > 150:
        return "Glas", "Glasscherbe"
    elif 10 < h < 30 and s > 50:
        return "Keramik", "Gefäßscherbe"
    elif s < 40 and v < 80:
        return "Metall", "Metallfragment"
    else:
        return "Unbekannt", "Fragment"


def generate_fragment_name(piece, api_key=None, use_ai=True):
    """Generiert einen Namen für das Fragment"""
    color = get_dominant_color(piece['thumbnail'], piece['mask'])
    material, obj_type = "Unbekannt", "Fragment"
    
    if use_ai and api_key:
        img_b64 = base64.b64encode(cv2.imencode('.png', piece['thumbnail'])[1]).decode()
        ai_result = analyze_fragment_with_ai(img_b64, api_key)
        if ai_result:
            material = ai_result.get('material', 'Unbekannt')
            obj_type = ai_result.get('object_type', 'Fragment')
            color = ai_result.get('color', color)
    else:
        material, obj_type = classify_material(piece)
    
    piece['material'] = material
    piece['object_type'] = obj_type
    piece['color_name'] = color
    
    return f"{material}_{obj_type}_{color}".replace(" ", "_")


def image_to_base64(img):
    """Konvertiert Bild zu Base64"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# =============================================================================
# SEGMENTIERUNG
# =============================================================================

def segment_fragments(image, min_area=100, excavation="", api_key=None, use_ai=True):
    """Segmentiert Fragmente aus einem Bild"""
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
        
        # Kontur relativ zum ROI
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
    """Extrahiert Features für Clustering"""
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
    """Generiert Farbe für Cluster-Visualisierung"""
    if cluster_id == -1:
        return "rgb(180, 180, 180)"
    hue = int((cluster_id * 47) % 180)
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


# =============================================================================
# PDF EXPORT
# =============================================================================

def generate_qr_code(data, size=10):
    qr = qrcode.QRCode(version=1, box_size=size, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def create_pdf(pieces, cluster_names=None, title="ShardMind"):
    if cluster_names is None:
        cluster_names = {}
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    cols, rows = 3, 4
    cell_w, cell_h = width / cols, height / rows
    
    active = [p for p in pieces if not p.get('deleted')]
    
    for idx, piece in enumerate(active):
        col = idx % cols
        row = (idx // cols) % rows
        
        if idx > 0 and idx % (cols * rows) == 0:
            c.showPage()
        
        x = col * cell_w + 15
        y = height - (row + 1) * cell_h + 15
        
        qr_img = generate_qr_code(f"shardmind://find/{piece['id']}", size=6)
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        c.drawImage(ImageReader(qr_buffer), x, y, width=70, height=70)
        
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 78, y + 58, piece['id'])
        c.setFont("Helvetica", 7)
        c.drawString(x + 78, y + 46, piece.get('name', 'N/A')[:20])
    
    c.save()
    buffer.seek(0)
    return buffer


# =============================================================================
# HILFE-INHALTE
# =============================================================================

HELP_CONTENT = {
    'de': {
        'overview': """
## 🏺 Was ist ShardMind?

ShardMind ist ein KI-gestütztes Werkzeug für die archäologische Fundbearbeitung. 
Es hilft dabei, zerbrochene Objekte (wie Keramik, Glas, etc.) zu analysieren und 
visuell zu rekonstruieren.

### Hauptfunktionen:

- **🔬 Automatische Segmentierung**: Erkennt einzelne Fragmente in Fotos
- **🤖 KI-Klassifikation**: Bestimmt Material und Objekttyp
- **📦 Intelligentes Clustering**: Gruppiert zusammengehörige Teile
- **🧩 Rekonstruktion**: Setzt Fragmente visuell zusammen
- **💾 Datenbank**: Speichert alle Funde mit Metadaten
- **🖨️ QR-Labels**: Erstellt druckbare Etiketten
""",
        'workflow': """
## 📋 Schritt-für-Schritt Anleitung

### 1. Fotografieren
- Lege die Fragmente auf einen **neutralen Hintergrund** (weiß, grau, schwarz)
- Sorge für **gleichmäßige Beleuchtung** ohne Schatten
- Die Teile sollten sich **nicht berühren**
- Fotografiere aus der **Vogelperspektive**

### 2. Hochladen & Analysieren
- Klicke auf "📤 Fotos hochladen"
- Wähle ein oder mehrere Bilder aus
- Stelle die Parameter ein (Min. Größe, Cluster-Sensitivität)
- Klicke auf "🔬 Analysieren"

### 3. Gruppen überprüfen
- Im Tab "📦 Gruppen" siehst du die automatisch erkannten Gruppen
- Teile mit ähnlichen Bruchkanten werden zusammen gruppiert
- Du kannst Gruppen umbenennen

### 4. Rekonstruieren
- Wähle eine Gruppe im Tab "🧩 Rekonstruktion"
- Klicke auf "Rekonstruktion berechnen"
- ShardMind analysiert die Bruchkanten und setzt die Teile zusammen
- Passe bei Bedarf manuell an

### 5. Speichern & Exportieren
- Speichere Gruppen in deiner Datenbank
- Exportiere Rekonstruktionen als Bild
- Erstelle PDF-Labels mit QR-Codes
""",
        'tips': """
## 💡 Tipps für beste Ergebnisse

### Fotografieren
- ✅ **Hohe Auflösung** (min. 1920x1080)
- ✅ **Neutraler Hintergrund** (Kontrast zu den Fragmenten)
- ✅ **Keine Schatten** (diffuses Licht verwenden)
- ✅ **Teile nicht überlappen** lassen
- ❌ Keine Reflexionen auf glänzenden Oberflächen

### Parameter
- **Min. Größe**: Erhöhen wenn zu viele kleine Artefakte erkannt werden
- **Cluster-Sensitivität**: Höher = weniger, größere Gruppen
- **Kanten-Gewichtung**: Höher = mehr Fokus auf Bruchkanten-Matching

### Rekonstruktion
- Beginne mit Gruppen die klar zusammengehören
- Nutze die manuelle Anpassung für Feintuning
- Exportiere Zwischenergebnisse

### API Key
- Mit Anthropic API Key: Präzisere Material-Erkennung
- Ohne API Key: Heuristische Klassifikation (funktioniert auch gut!)
""",
        'faq': """
## ❓ Häufige Fragen

**Q: Warum werden meine Fragmente nicht erkannt?**
> A: Prüfe den Hintergrund-Kontrast und die Min. Größe-Einstellung.

**Q: Warum sind Teile in falschen Gruppen?**
> A: Passe die Cluster-Sensitivität an. Höherer Wert = strengere Gruppierung.

**Q: Was ist der Unterschied mit/ohne API Key?**
> A: Mit API Key nutzt ShardMind Claude Vision für präzisere Materialerkennung. 
> Ohne API Key wird eine heuristische Methode verwendet.

**Q: Kann ich meine Daten exportieren?**
> A: Ja! Rekonstruktionen als Bild, Labels als PDF.

**Q: Sind meine Daten sicher?**
> A: Jeder Benutzer hat eine eigene, passwortgeschützte Datenbank.

**Q: Welche Materialien werden erkannt?**
> A: Keramik, Glas, Knochen, Metall, Stein und mehr.
"""
    },
    'en': {
        'overview': """
## 🏺 What is ShardMind?

ShardMind is an AI-powered tool for archaeological fragment analysis. 
It helps analyze broken objects (ceramics, glass, etc.) and visually reconstruct them.

### Main Features:

- **🔬 Automatic Segmentation**: Detects individual fragments in photos
- **🤖 AI Classification**: Determines material and object type
- **📦 Smart Clustering**: Groups related pieces together
- **🧩 Reconstruction**: Visually assembles fragments
- **💾 Database**: Stores all finds with metadata
- **🖨️ QR Labels**: Creates printable labels
""",
        'workflow': """
## 📋 Step-by-Step Guide

### 1. Photographing
- Place fragments on a **neutral background** (white, gray, black)
- Ensure **even lighting** without shadows
- Pieces should **not touch** each other
- Photograph from **bird's eye view**

### 2. Upload & Analyze
- Click "📤 Upload Photos"
- Select one or more images
- Adjust parameters (Min. Size, Cluster Sensitivity)
- Click "🔬 Analyze"

### 3. Review Groups
- In the "📦 Groups" tab you see automatically detected groups
- Pieces with similar break edges are grouped together
- You can rename groups

### 4. Reconstruct
- Select a group in the "🧩 Reconstruction" tab
- Click "Calculate Reconstruction"
- ShardMind analyzes break edges and assembles pieces
- Adjust manually if needed

### 5. Save & Export
- Save groups to your database
- Export reconstructions as images
- Create PDF labels with QR codes
""",
        'tips': """
## 💡 Tips for Best Results

### Photography
- ✅ **High resolution** (min. 1920x1080)
- ✅ **Neutral background** (contrast with fragments)
- ✅ **No shadows** (use diffuse light)
- ✅ **Don't overlap pieces**
- ❌ No reflections on shiny surfaces

### Parameters
- **Min. Size**: Increase if too many small artifacts detected
- **Cluster Sensitivity**: Higher = fewer, larger groups
- **Edge Weight**: Higher = more focus on break edge matching

### Reconstruction
- Start with groups that clearly belong together
- Use manual adjustment for fine-tuning
- Export intermediate results

### API Key
- With Anthropic API Key: More precise material detection
- Without API Key: Heuristic classification (also works well!)
""",
        'faq': """
## ❓ Frequently Asked Questions

**Q: Why aren't my fragments detected?**
> A: Check background contrast and Min. Size setting.

**Q: Why are pieces in wrong groups?**
> A: Adjust Cluster Sensitivity. Higher value = stricter grouping.

**Q: What's the difference with/without API Key?**
> A: With API Key, ShardMind uses Claude Vision for precise material detection. 
> Without API Key, a heuristic method is used.

**Q: Can I export my data?**
> A: Yes! Reconstructions as images, labels as PDF.

**Q: Is my data secure?**
> A: Each user has their own password-protected database.

**Q: Which materials are detected?**
> A: Ceramics, glass, bone, metal, stone and more.
"""
    }
}


# =============================================================================
# STREAMLIT APP
# =============================================================================

def show_login_page():
    """Zeigt die Login/Registrierungs-Seite"""
    st.title(f"🏺 {t('app_title')}")
    st.caption(t('app_subtitle'))
    
    # Sprache wählen
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        lang = st.selectbox(
            t('language'),
            ['de', 'en'],
            format_func=lambda x: '🇩🇪 Deutsch' if x == 'de' else '🇬🇧 English',
            key='lang_select_login'
        )
        if lang != st.session_state.get('language', 'de'):
            st.session_state.language = lang
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs([f"🔐 {t('login')}", f"📝 {t('register')}"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input(t('username'))
            password = st.text_input(t('password'), type='password')
            submitted = st.form_submit_button(t('login_btn'), use_container_width=True)
            
            if submitted:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(t('login_error'))
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input(t('username'), key='reg_user')
            new_password = st.text_input(t('password'), type='password', key='reg_pass')
            confirm_password = st.text_input(t('password_confirm'), type='password')
            submitted = st.form_submit_button(t('register_btn'), use_container_width=True)
            
            if submitted:
                if new_password != confirm_password:
                    st.error(t('register_error_password'))
                else:
                    success, msg = register_user(new_username, new_password)
                    if success:
                        st.success(t(msg))
                    else:
                        st.error(t(msg))


def main():
    st.set_page_config(
        page_title=f"ShardMind v{APP_VERSION}",
        page_icon="🏺",
        layout="wide"
    )
    
    # Session State initialisieren
    if 'language' not in st.session_state:
        st.session_state.language = 'de'
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'pieces' not in st.session_state:
        st.session_state.pieces = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Login prüfen
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # Benutzer-Datenbank laden
    username = st.session_state.username
    db = get_user_database(username)
    
    # SIDEBAR
    with st.sidebar:
        st.title(f"🏺 {t('app_title')}")
        st.caption(f"v{APP_VERSION}")
        
        # Benutzer-Info
        st.success(f"👤 {t('logged_in_as')}: **{username}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t('logout_btn'), use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.pieces = []
                st.rerun()
        with col2:
            lang = st.selectbox(
                "",
                ['de', 'en'],
                format_func=lambda x: '🇩🇪' if x == 'de' else '🇬🇧',
                index=0 if st.session_state.language == 'de' else 1,
                key='lang_select_main'
            )
            if lang != st.session_state.language:
                st.session_state.language = lang
                st.rerun()
        
        st.markdown("---")
        
        # API
        with st.expander(t('api_settings'), expanded=False):
            api_key = st.text_input(t('api_key'), value=st.session_state.api_key, type='password')
            st.session_state.api_key = api_key
            use_ai = st.checkbox(t('use_ai'), value=bool(api_key), disabled=not api_key)
        
        st.markdown("---")
        
        # Upload
        files = st.file_uploader(t('upload_photos'), type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        st.markdown("---")
        
        # Parameter
        st.subheader(t('parameters'))
        min_area = st.slider(t('min_size'), 50, 1000, 200)
        cluster_threshold = st.slider(t('cluster_sens'), 10, 90, 40)
        edge_weight = st.slider(t('edge_weight'), 0.0, 1.0, 0.6, 0.1)
        
        excavation = st.text_input(t('excavation'), value=f"Excavation_{datetime.now().strftime('%Y')}")
        
        st.markdown("---")
        
        # Analyse starten
        if st.button(t('analyze_btn'), type='primary', use_container_width=True):
            if files:
                with st.spinner(t('analyzing_msg')):
                    all_pieces = []
                    progress = st.progress(0)
                    
                    for i, f in enumerate(files):
                        img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        pieces = segment_fragments(img, min_area, excavation, 
                                                   st.session_state.api_key if use_ai else None, use_ai)
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
            st.session_state.cluster_names = {}
            st.rerun()
        
        st.markdown("---")
        st.metric(t('fragments'), len(db.get('pieces', {})))
        st.metric(t('groups'), len(db.get('clusters', {})))
    
    # HAUPTBEREICH
    if st.session_state.pieces:
        active_pieces = [p for p in st.session_state.pieces if not p.get('deleted')]
        
        # Kanten-basiertes Clustering
        if len(active_pieces) > 1:
            labels = cluster_by_reconstruction_potential(
                active_pieces, 
                edge_weight=edge_weight,
                color_weight=1-edge_weight,
                threshold=cluster_threshold
            )
            for i, p in enumerate(active_pieces):
                p['cluster'] = labels[i]
        else:
            for p in active_pieces:
                p['cluster'] = -1
        
        cluster_ids = set([p.get('cluster', -1) for p in active_pieces])
        n_clusters = len([c for c in cluster_ids if c >= 0])
        
        # Metriken
        col1, col2, col3 = st.columns(3)
        col1.metric(f"🏺 {t('fragments')}", len(active_pieces))
        col2.metric(f"📦 {t('groups')}", n_clusters)
        col3.metric(f"🗺️ {t('excavation')}", excavation[:15])
        
        st.markdown("---")
        
        # TABS
        tabs = st.tabs([
            t('tab_gallery'),
            t('tab_groups'),
            t('tab_reconstruction'),
            t('tab_database'),
            t('tab_help')
        ])
        
        # TAB 1: GALERIE
        with tabs[0]:
            st.header(t('detected_fragments'))
            
            if st.button(t('recluster_btn')):
                labels = cluster_by_reconstruction_potential(active_pieces, edge_weight, 1-edge_weight, cluster_threshold)
                for i, p in enumerate(active_pieces):
                    p['cluster'] = labels[i]
                st.rerun()
            
            cols = st.columns(5)
            for i, p in enumerate(active_pieces):
                with cols[i % 5]:
                    color = get_cluster_color(p.get('cluster', -1))
                    st.markdown(
                        f'<div style="border: 3px solid {color}; padding: 3px; border-radius: 5px;">'
                        f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;">'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"**{p['id'][:11]}**")
                    st.caption(f"{p.get('name', 'N/A')[:20]}")
        
        # TAB 2: GRUPPEN
        with tabs[1]:
            st.header(t('groups_title'))
            
            cluster_ids_sorted = sorted([c for c in cluster_ids if c >= 0])
            
            if not cluster_ids_sorted:
                st.warning(t('no_groups'))
                st.info(f"💡 {t('increase_sensitivity')}")
            else:
                for cluster_id in cluster_ids_sorted:
                    cluster_pieces = [p for p in active_pieces if p.get('cluster') == cluster_id]
                    
                    materials = [p.get('material', 'Unknown') for p in cluster_pieces]
                    common_mat = max(set(materials), key=materials.count)
                    default_name = st.session_state.cluster_names.get(cluster_id, f"{common_mat}_Group_{cluster_id}")
                    
                    with st.expander(f"📦 {default_name} ({len(cluster_pieces)} {t('pieces')})", expanded=True):
                        cluster_name = st.text_input(t('group_name'), value=default_name, key=f"name_{cluster_id}")
                        st.session_state.cluster_names[cluster_id] = cluster_name
                        
                        preview_cols = st.columns(min(6, len(cluster_pieces)))
                        for i, p in enumerate(cluster_pieces[:6]):
                            with preview_cols[i]:
                                st.image(p['thumbnail'])
                                st.caption(p['id'][:8])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(t('save_btn'), key=f"save_{cluster_id}"):
                                cluster_key = f"{cluster_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                db['clusters'][cluster_key] = {
                                    'name': cluster_name,
                                    'created': datetime.now().isoformat(),
                                    'piece_ids': [p['id'] for p in cluster_pieces]
                                }
                                for p in cluster_pieces:
                                    db['pieces'][p['id']] = p.copy()
                                save_user_database(username, db)
                                st.success(f"✓ {t('saved')}")
                        
                        with col2:
                            if st.button(t('reconstruct_btn'), key=f"recon_{cluster_id}", type="primary"):
                                st.session_state.recon_cluster = cluster_id
        
        # TAB 3: REKONSTRUKTION
        with tabs[2]:
            st.header(t('reconstruction_title'))
            
            st.info(t('reconstruction_info'))
            
            cluster_options = {
                f"{st.session_state.cluster_names.get(c, f'Group_{c}')} ({sum(1 for p in active_pieces if p.get('cluster') == c)} {t('pieces')})": c
                for c in cluster_ids if c >= 0
            }
            
            if cluster_options:
                selected_name = st.selectbox(t('select_group'), list(cluster_options.keys()))
                selected_id = cluster_options[selected_name]
                
                recon_pieces = [p for p in active_pieces if p.get('cluster') == selected_id]
                
                st.markdown(f"### {len(recon_pieces)} {t('fragments_in_group')}")
                
                preview_cols = st.columns(min(8, len(recon_pieces)))
                for i, p in enumerate(recon_pieces[:8]):
                    with preview_cols[i]:
                        st.image(p['thumbnail'], use_container_width=True)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    canvas_size = st.slider(t('canvas_size'), 400, 1200, 800, 100)
                with col2:
                    show_edges = st.checkbox(t('show_edges'), value=True)
                
                if st.button(t('calculate_btn'), type='primary', use_container_width=True):
                    with st.spinner(t('analyzing')):
                        recon_img, placements, matches = reconstruct_group(recon_pieces, canvas_size)
                        
                        if recon_img is not None:
                            st.session_state.recon_image = recon_img
                            st.session_state.recon_placements = placements
                            st.session_state.recon_matches = matches
                            st.success(f"✓ {len(matches)} {t('matches_found')}")
                
                if 'recon_image' in st.session_state and st.session_state.recon_image is not None:
                    st.markdown(f"### {t('result_title')}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        recon_rgb = cv2.cvtColor(st.session_state.recon_image, cv2.COLOR_BGR2RGB)
                        st.image(recon_rgb, use_container_width=True)
                        
                        img_buffer = io.BytesIO()
                        Image.fromarray(recon_rgb).save(img_buffer, format='PNG')
                        st.download_button(
                            t('export_btn'),
                            data=img_buffer.getvalue(),
                            file_name=f"reconstruction_{selected_id}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        st.markdown(f"#### {t('edge_matches')}")
                        for i, match in enumerate(st.session_state.get('recon_matches', [])[:5]):
                            st.write(f"**{t('match')} {i+1}:** {match['score']:.1f}%")
                        
                        st.markdown("---")
                        st.markdown(f"#### {t('manual_adjust')}")
                        
                        if recon_pieces:
                            piece_idx = st.selectbox(
                                t('select_fragment'),
                                range(len(recon_pieces)),
                                format_func=lambda x: recon_pieces[x]['id'][:11]
                            )
                            
                            if piece_idx < len(st.session_state.get('recon_placements', [])):
                                placement = st.session_state.recon_placements[piece_idx]
                                
                                new_x = st.slider(t('x_position'), 0, canvas_size, int(placement.get('x', 400)))
                                new_y = st.slider(t('y_position'), 0, canvas_size, int(placement.get('y', 400)))
                                new_rot = st.slider(t('rotation'), -180, 180, int(np.degrees(placement.get('rotation', 0))))
                                new_scale = st.slider(t('scale'), 0.5, 2.0, placement.get('scale', 1.0), 0.1)
                                
                                if st.button(t('apply_btn')):
                                    st.session_state.recon_placements[piece_idx] = {
                                        'x': new_x, 'y': new_y,
                                        'rotation': np.radians(new_rot),
                                        'scale': new_scale
                                    }
                                    recon_img, _, _ = reconstruct_group(recon_pieces, canvas_size)
                                    st.session_state.recon_image = recon_img
                                    st.rerun()
                
                if show_edges and recon_pieces:
                    st.markdown(f"### {t('edge_analysis')}")
                    edge_cols = st.columns(min(4, len(recon_pieces)))
                    for i, p in enumerate(recon_pieces[:4]):
                        with edge_cols[i]:
                            vis = p['thumbnail'].copy()
                            if 'contour' in p:
                                cv2.drawContours(vis, [p['contour']], -1, (0, 255, 0), 2)
                            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{t('edges')}: {p['id'][:8]}")
            else:
                st.warning(t('no_groups_warning'))
        
        # TAB 4: DATENBANK
        with tabs[3]:
            st.header(t('database_title'))
            
            if db.get('pieces'):
                st.write(f"**{len(db['pieces'])} {t('fragments_saved')}**")
                
                cols = st.columns(6)
                for i, (pid, piece) in enumerate(list(db['pieces'].items())[:12]):
                    with cols[i % 6]:
                        if 'thumbnail' in piece:
                            st.image(piece['thumbnail'])
                        st.caption(pid[:11])
                
                st.markdown("---")
                
                with st.expander(t('delete_db')):
                    confirm = st.text_input(t('delete_confirm'))
                    delete_word = "LÖSCHEN" if st.session_state.language == 'de' else "DELETE"
                    if st.button(t('delete_btn'), type='secondary'):
                        if confirm == delete_word:
                            db['pieces'] = {}
                            db['clusters'] = {}
                            save_user_database(username, db)
                            st.success("✓")
                            st.rerun()
            else:
                st.info(t('db_empty'))
            
            if db.get('clusters'):
                st.markdown("---")
                st.write(f"**{t('groups_saved')}**")
                for key, cluster in db['clusters'].items():
                    st.write(f"• {cluster['name']} ({len(cluster.get('piece_ids', []))} {t('pieces')})")
        
        # TAB 5: HILFE
        with tabs[4]:
            st.header(t('help_title'))
            
            lang = st.session_state.language
            help_tabs = st.tabs([t('help_overview'), t('help_workflow'), t('help_tips'), t('help_faq')])
            
            with help_tabs[0]:
                st.markdown(HELP_CONTENT[lang]['overview'])
            
            with help_tabs[1]:
                st.markdown(HELP_CONTENT[lang]['workflow'])
            
            with help_tabs[2]:
                st.markdown(HELP_CONTENT[lang]['tips'])
            
            with help_tabs[3]:
                st.markdown(HELP_CONTENT[lang]['faq'])
    
    else:
        # Startseite
        st.title(f"🏺 {t('app_title')} v{APP_VERSION}")
        st.markdown(f"### {t('app_subtitle')}")
        
        # Zeige Hilfe auf der Startseite
        lang = st.session_state.language
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(HELP_CONTENT[lang]['overview'])
        
        with col2:
            st.markdown(HELP_CONTENT[lang]['tips'])


if __name__ == "__main__":
    main()
