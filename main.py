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
║   Version 1.0                                                                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG:
==========

Version 1.0 (2025-01-11)
------------------------
NEUE FEATURES:
• Eindeutige UUIDs für alle Fundstücke (statt fortlaufender Nummern)
• Verbesserte QR-Codes mit UUID-basierter Identifikation
• PDF-Export für gesamte Datenbank oder einzelne Fundstücke
• Archäologie-fokussierte Benutzeroberfläche
• KI-gestützte Objekterkennung via Claude Vision API
• Automatische Klassifikation: Keramik, Glas, Knochen, Metall, etc.
• Fundort-Tracking mit Grabungskontext
• Verbesserte Cluster-Benennung im PDF-Export

VERBESSERUNGEN:
• Professionelle archäologische Terminologie
• Erweiterte Metadaten pro Fundstück (Fundort, Schicht, Material)
• Batch-System für Grabungskampagnen
• Such- und Filterfunktionen in der Datenbank

BEKANNTE EINSCHRÄNKUNGEN:
• KI-Erkennung benötigt Anthropic API Key
• Ohne API: Heuristische Materialerkennung

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
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
import requests
import json
import os
import uuid
import hashlib

# =============================================================================
# KONFIGURATION
# =============================================================================

APP_VERSION = "1.0"
DB_PATH = Path("shardmind_database_v1.pkl")
FEATURE_VERSION = 10

# Archäologische Materialtypen
MATERIAL_TYPES = [
    "Keramik", "Glas", "Knochen", "Metall", "Stein", 
    "Holz", "Textil", "Leder", "Bernstein", "Unbekannt"
]

# Archäologische Perioden
PERIODS = [
    "Unbestimmt", "Neolithikum", "Bronzezeit", "Eisenzeit",
    "Römisch", "Mittelalter", "Neuzeit", "Modern"
]


# =============================================================================
# EINDEUTIGE ID-GENERIERUNG
# =============================================================================

def generate_unique_id():
    """
    Generiert eine eindeutige ID für Fundstücke.
    Format: SM-XXXXXXXX (SM = ShardMind, 8 Zeichen aus UUID)
    """
    unique = uuid.uuid4().hex[:8].upper()
    return f"SM-{unique}"


def generate_short_hash(data):
    """Generiert einen kurzen Hash für Duplikat-Erkennung"""
    return hashlib.md5(str(data).encode()).hexdigest()[:8]


# =============================================================================
# KI-OBJEKTERKENNUNG
# =============================================================================

def analyze_fragment_with_ai(image_base64, api_key=None):
    """
    Nutzt Claude API um das archäologische Fragment zu analysieren.
    Gibt zurück: {"material": "Keramik", "object_type": "Gefäßscherbe", "color": "Rotbraun", "notes": "..."}
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
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
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Du bist ein Archäologe. Analysiere dieses Bild eines Fundstücks/Fragments.
Antworte NUR mit einem JSON-Objekt (keine anderen Texte):
{"material": "Materialtyp", "object_type": "Objekttyp", "color": "Hauptfarbe", "condition": "Zustand"}

Materialtypen: Keramik, Glas, Knochen, Metall, Stein, Holz, Textil, Unbekannt
Objekttypen für Keramik: Gefäßscherbe, Randscherbe, Bodenscherbe, Henkelscherbe, Wandungsscherbe
Objekttypen für Glas: Glasscherbe, Flaschenfragment, Fensterglas
Objekttypen für Knochen: Knochenfragment, Zahn, Geweih
Objekttypen für Metall: Münze, Fibel, Nagel, Beschlag, Metallfragment
Objekttypen für Stein: Feuerstein, Mahlstein, Steinwerkzeug

Zustand: Gut, Fragmentiert, Verwittert, Korrodiert, Beschädigt

Wenn unklar, sage "Unbekanntes_Fragment"."""
                            }
                        ]
                    }
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['content'][0]['text']
            try:
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text.strip())
            except:
                return None
        return None
    except Exception as e:
        print(f"API Error: {e}")
        return None


def get_dominant_color_name(image, mask):
    """Extrahiert die dominante Farbe und gibt einen archäologischen Farbnamen zurück"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
    if s < 30:
        if v < 60:
            return "Schwarz"
        elif v < 120:
            return "Dunkelgrau"
        elif v < 180:
            return "Grau"
        else:
            return "Weiß"
    elif h < 10 or h > 170:
        if v < 100:
            return "Dunkelrot"
        else:
            return "Rotbraun"
    elif h < 25:
        if s > 150:
            return "Orange"
        else:
            return "Terrakotta"
    elif h < 40:
        return "Ocker"
    elif h < 75:
        if v < 100:
            return "Dunkelgrün"
        else:
            return "Grün"
    elif h < 130:
        if v < 100:
            return "Dunkelblau"
        else:
            return "Blau"
    else:
        return "Violett"


def classify_material_heuristic(piece):
    """
    Heuristische Materialklassifikation basierend auf Bildmerkmalen.
    Fallback wenn keine API verfügbar.
    """
    area = piece['area']
    perimeter = cv2.arcLength(piece['contour'], True)
    compactness = (perimeter ** 2) / (area + 1e-6)
    
    # Farbanalyse
    hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=piece['mask'])
    h, s, v = mean_hsv[:3]
    
    # Konvexität
    hull = cv2.convexHull(piece['contour'])
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    
    # Materialbestimmung basierend auf Merkmalen
    if s < 20 and v > 150:  # Sehr hell, wenig Sättigung -> Glas oder Knochen
        if solidity > 0.8:
            return "Knochen", "Knochenfragment"
        else:
            return "Glas", "Glasscherbe"
    
    elif 10 < h < 30 and s > 50:  # Rot-Orange-Braun -> Keramik
        if solidity > 0.85:
            return "Keramik", "Wandungsscherbe"
        elif compactness > 30:
            return "Keramik", "Randscherbe"
        else:
            return "Keramik", "Gefäßscherbe"
    
    elif s < 40 and v < 80:  # Dunkel, wenig Farbe -> Metall
        return "Metall", "Metallfragment"
    
    elif s < 30 and 80 < v < 150:  # Grau -> Stein
        return "Stein", "Steinfragment"
    
    else:
        return "Unbekannt", "Fragment"


def generate_fragment_name(piece, api_key=None, use_ai=True):
    """
    Generiert einen archäologischen Namen für das Fundstück.
    Format: Material_Objekttyp_Farbe (z.B. "Keramik_Randscherbe_Rotbraun")
    """
    color = get_dominant_color_name(piece['thumbnail'], piece['mask'])
    material = "Unbekannt"
    object_type = "Fragment"
    condition = "Unbestimmt"
    
    # Versuche KI-Erkennung
    if use_ai and api_key:
        img_base64 = image_to_base64_raw(piece['thumbnail'])
        ai_result = analyze_fragment_with_ai(img_base64, api_key)
        
        if ai_result:
            material = ai_result.get('material', 'Unbekannt').replace(" ", "_")
            object_type = ai_result.get('object_type', 'Fragment').replace(" ", "_")
            if 'color' in ai_result and ai_result['color']:
                color = ai_result['color'].replace(" ", "_")
            if 'condition' in ai_result:
                condition = ai_result['condition']
    else:
        # Fallback: Heuristische Klassifikation
        material, object_type = classify_material_heuristic(piece)
    
    piece['material'] = material
    piece['object_type'] = object_type
    piece['color_name'] = color
    piece['condition'] = condition
    
    return f"{material}_{object_type}_{color}"


def image_to_base64_raw(img):
    """Konvertiert OpenCV-Bild zu Base64"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def image_to_base64(img):
    """Konvertiert OpenCV-Bild zu Base64 für HTML-Anzeige"""
    return image_to_base64_raw(img)


# =============================================================================
# DATENBANK
# =============================================================================

def load_database():
    if DB_PATH.exists():
        try:
            with open(DB_PATH, 'rb') as f:
                db = pickle.load(f)
                if db.get('version') != FEATURE_VERSION:
                    # Migration
                    return migrate_database(db)
                return db
        except:
            return create_empty_database()
    return create_empty_database()


def create_empty_database():
    return {
        'pieces': {},  # Dict mit UUID als Key
        'clusters': {},
        'excavations': {},  # Grabungskampagnen
        'version': FEATURE_VERSION,
        'created': datetime.now().isoformat(),
        'app_version': APP_VERSION
    }


def migrate_database(old_db):
    """Migriert alte Datenbank-Formate"""
    new_db = create_empty_database()
    
    # Alte Teile migrieren
    if 'pieces' in old_db:
        if isinstance(old_db['pieces'], list):
            for p in old_db['pieces']:
                new_id = generate_unique_id()
                p['id'] = new_id
                new_db['pieces'][new_id] = p
        elif isinstance(old_db['pieces'], dict):
            new_db['pieces'] = old_db['pieces']
    
    if 'clusters' in old_db:
        new_db['clusters'] = old_db['clusters']
    
    return new_db


def save_database(db):
    db['version'] = FEATURE_VERSION
    db['last_modified'] = datetime.now().isoformat()
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)


# =============================================================================
# QR-CODE & PDF GENERATION
# =============================================================================

def generate_qr_code(data, size=10):
    """Generiert QR-Code als PIL Image"""
    qr = qrcode.QRCode(
        version=1, 
        box_size=size, 
        border=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M
    )
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def create_single_label_pdf(piece, cluster_name=None):
    """Erstellt PDF-Label für ein einzelnes Fundstück"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(200, 280))
    
    # QR-Code
    qr_data = f"shardmind://find/{piece['id']}"
    qr_img = generate_qr_code(qr_data, size=8)
    qr_buffer = io.BytesIO()
    qr_img.save(qr_buffer, format='PNG')
    qr_buffer.seek(0)
    c.drawImage(ImageReader(qr_buffer), 10, 160, width=100, height=100)
    
    # Fundstück-Bild
    if 'thumbnail' in piece:
        thumb_buffer = io.BytesIO()
        img_rgb = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((80, 80))
        pil_img.save(thumb_buffer, format='PNG')
        thumb_buffer.seek(0)
        c.drawImage(ImageReader(thumb_buffer), 115, 180, width=75, height=75)
    
    # Text
    c.setFont("Helvetica-Bold", 10)
    c.drawString(10, 145, f"ID: {piece['id']}")
    
    c.setFont("Helvetica", 8)
    y = 130
    
    name = piece.get('name', 'Unbenannt')
    if len(name) > 25:
        name = name[:22] + "..."
    c.drawString(10, y, f"Name: {name}")
    y -= 12
    
    if piece.get('material'):
        c.drawString(10, y, f"Material: {piece['material']}")
        y -= 12
    
    if piece.get('excavation'):
        exc = piece['excavation'][:20] if len(piece['excavation']) > 20 else piece['excavation']
        c.drawString(10, y, f"Grabung: {exc}")
        y -= 12
    
    if cluster_name:
        cl = cluster_name[:20] if len(cluster_name) > 20 else cluster_name
        c.drawString(10, y, f"Gruppe: {cl}")
        y -= 12
    
    # Datum
    c.setFont("Helvetica", 6)
    c.drawString(10, 10, f"Erstellt: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(10, 3, f"ShardMind v{APP_VERSION}")
    
    c.save()
    buffer.seek(0)
    return buffer


def create_multi_label_pdf(pieces, cluster_names_map=None, title="ShardMind Fundstück-Labels"):
    """
    Erstellt PDF mit mehreren QR-Code-Labels.
    Layout: 3 Spalten, 4 Zeilen pro Seite (größere Labels für bessere Lesbarkeit)
    """
    if cluster_names_map is None:
        cluster_names_map = {}
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Layout: 3 Spalten, 4 Zeilen
    cols, rows = 3, 4
    cell_width = width / cols
    cell_height = height / rows
    
    margin = 15
    qr_size = 70
    
    # Titel auf erster Seite
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - 30, title)
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - 45, f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Anzahl: {len(pieces)}")
    
    # Startposition (nach Titel)
    start_row_offset = 60
    
    active_pieces = [p for p in pieces if not p.get('deleted', False)]
    
    for idx, piece in enumerate(active_pieces):
        # Position berechnen
        col = idx % cols
        row = (idx // cols) % rows
        page_num = idx // (cols * rows)
        
        if idx > 0 and idx % (cols * rows) == 0:
            c.showPage()
            start_row_offset = 0
        
        x = col * cell_width + margin
        y = height - (row + 1) * cell_height - start_row_offset + margin
        
        # QR-Code mit eindeutiger ID
        qr_data = f"shardmind://find/{piece['id']}"
        if piece.get('excavation'):
            qr_data += f"?exc={piece['excavation'][:20]}"
        
        qr_img = generate_qr_code(qr_data, size=6)
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        c.drawImage(ImageReader(qr_buffer), x, y, width=qr_size, height=qr_size)
        
        # Text rechts vom QR-Code
        text_x = x + qr_size + 8
        text_y = y + qr_size - 12
        
        # ID (eindeutig!)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(text_x, text_y, piece['id'])
        text_y -= 11
        
        # Name
        c.setFont("Helvetica", 7)
        name = piece.get('name', 'Unbenannt')
        if len(name) > 18:
            name = name[:15] + "..."
        c.drawString(text_x, text_y, name)
        text_y -= 10
        
        # Material
        if piece.get('material'):
            c.setFont("Helvetica", 6)
            c.drawString(text_x, text_y, f"Mat: {piece['material'][:12]}")
            text_y -= 9
        
        # Grabung/Batch
        if piece.get('excavation'):
            c.setFont("Helvetica", 6)
            exc = piece['excavation'][:15] if len(piece['excavation']) > 15 else piece['excavation']
            c.drawString(text_x, text_y, f"Grab: {exc}")
            text_y -= 9
        
        # Cluster-Name
        cluster_id = piece.get('cluster', -1)
        if cluster_id >= 0:
            cluster_name = cluster_names_map.get(cluster_id, f"Gruppe_{cluster_id}")
            c.setFont("Helvetica", 6)
            if len(cluster_name) > 15:
                cluster_name = cluster_name[:12] + "..."
            c.drawString(text_x, text_y, f"Grp: {cluster_name}")
        
        # Rahmen (gestrichelt für Schnittlinie)
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(0.3)
        c.setDash(3, 3)
        c.rect(x - 5, y - 8, cell_width - 10, cell_height - 15)
    
    # Footer auf letzter Seite
    c.setDash()
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(margin, 15, f"ShardMind v{APP_VERSION} - Archäologische Scherben-Analyse")
    
    c.save()
    buffer.seek(0)
    return buffer


def create_database_export_pdf(db, cluster_names_map=None):
    """Erstellt PDF-Export der gesamten Datenbank"""
    all_pieces = list(db['pieces'].values())
    return create_multi_label_pdf(
        all_pieces, 
        cluster_names_map,
        title=f"ShardMind Datenbank-Export ({len(all_pieces)} Fundstücke)"
    )


# =============================================================================
# BILDVERARBEITUNG & SEGMENTIERUNG
# =============================================================================

def is_valid_fragment(contour, roi, mask_roi, image_shape):
    """Prüft ob eine Kontur ein valides Fundstück sein könnte"""
    area = cv2.contourArea(contour)
    if area < 200 or area > (image_shape[0] * image_shape[1] * 0.85):
        return False
    
    x, y, wb, hb = cv2.boundingRect(contour)
    aspect = wb / (hb + 1e-6)
    if aspect < 0.08 or aspect > 12:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if (perimeter ** 2) / (area + 1e-6) > 120:
        return False
    
    if roi.shape[0] > 5 and roi.shape[1] > 5:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        _, std = cv2.meanStdDev(lab, mask=mask_roi)
        if np.all(std < 5):
            return False
        mean_val = cv2.mean(lab, mask=mask_roi)
        if mean_val[0] > 240 and abs(mean_val[1] - 128) < 10 and abs(mean_val[2] - 128) < 10:
            return False
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if area / (hull_area + 1e-6) < 0.5:
        return False
    
    return True


def segment_fragments(image, min_area=100, excavation="", api_key=None, use_ai=True, progress_callback=None):
    """
    Segmentiert Fundstücke aus einem Bild.
    Jedes Fundstück erhält eine eindeutige UUID.
    """
    h, w = image.shape[:2]
    pad = 30
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    blurred = cv2.GaussianBlur(padded, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh = cv2.bitwise_or(cv2.bitwise_or(thresh1, thresh2), thresh3)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=5)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        x, y, wb, hb = cv2.boundingRect(c)
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(padded.shape[1], x + wb + margin), min(padded.shape[0], y + hb + margin)
        
        roi = padded[y1:y2, x1:x2].copy()
        mask = np.zeros(padded.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask_roi = mask[y1:y2, x1:x2].copy()
        
        if not is_valid_fragment(c, roi, mask_roi, padded.shape):
            continue
        
        valid_contours.append({
            'contour': c,
            'roi': roi,
            'mask_roi': mask_roi,
            'area': cv2.contourArea(c)
        })
    
    pieces = []
    for i, vc in enumerate(valid_contours):
        # Eindeutige ID generieren
        unique_id = generate_unique_id()
        
        piece = {
            'id': unique_id,
            'contour': vc['contour'],
            'thumbnail': vc['roi'],
            'mask': vc['mask_roi'],
            'area': vc['area'],
            'deleted': False,
            'excavation': excavation,
            'created': datetime.now().isoformat(),
            'layer': "",  # Grabungsschicht
            'notes': ""
        }
        
        # Name generieren
        piece['name'] = generate_fragment_name(piece, api_key, use_ai)
        pieces.append(piece)
        
        if progress_callback:
            progress_callback(i + 1, len(valid_contours))
    
    return pieces


# =============================================================================
# FEATURE-EXTRAKTION & CLUSTERING
# =============================================================================

def get_features(p):
    """Extrahiert Features für Matching und Clustering"""
    M = cv2.moments(p['contour'])
    if M['m00'] == 0:
        return None
    
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    peri = cv2.arcLength(p['contour'], True)
    approx = cv2.approxPolyDP(p['contour'], 0.004 * peri, True)
    pts = approx.squeeze().astype(float)
    
    if pts.ndim == 1 or len(pts) < 3:
        return None
    
    diff = pts - np.array([cx, cy])
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    s_idx = np.argsort(angles)
    
    sig = np.interp(np.linspace(-np.pi, np.pi, 180), angles[s_idx], dists[s_idx], period=2 * np.pi)
    sig = sig / (sig.max() + 1e-6)

    lab = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2HSV)
    lab_mean, lab_std = cv2.meanStdDev(lab, mask=p['mask'])
    hsv_mean, hsv_std = cv2.meanStdDev(hsv, mask=p['mask'])
    
    color_features = np.concatenate([lab_mean.flatten(), lab_std.flatten(), hsv_mean.flatten(), hsv_std.flatten()])
    
    return {'shape': sig, 'color': color_features}


def calculate_match_score(f1, f2):
    """Berechnet Ähnlichkeitsscore zwischen zwei Fundstücken"""
    dist_c = np.linalg.norm(f1['color'] - f2['color'])
    score_c = max(0, 100 - (dist_c / 3.0))
    
    s1, s2 = f1['shape'], f2['shape']
    best_diff = min([np.mean((s1 - np.roll(s2, r)) ** 2) for r in range(0, 180, 3)])
    
    score_s = 95 + (0.02 - best_diff) * 250 if best_diff < 0.02 else max(0, 95 * (1 - best_diff * 6))
    
    return min(100, max(0, (0.4 * score_c) + (0.6 * score_s)))


def cluster_fragments(active_pieces, distance_threshold=1.5):
    """Clustert Fundstücke nach Ähnlichkeit"""
    if len(active_pieces) < 2:
        return [-1] * len(active_pieces)
    
    X = np.array([p['features']['color'] for p in active_pieces])
    X_scaled = StandardScaler().fit_transform(X)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='euclidean'
    )
    
    return clustering.fit_predict(X_scaled)


def get_cluster_color(cluster_id):
    """Generiert eine Farbe für Cluster-Visualisierung"""
    if cluster_id == -1:
        return "rgb(180, 180, 180)"
    hue = int((cluster_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title=f"ShardMind v{APP_VERSION} - Archäologische Analyse", 
        page_icon="🏺",
        layout="wide"
    )

    # Session State
    if 'pieces' not in st.session_state:
        st.session_state.pieces = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}
    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    db = load_database()

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.title("🏺 ShardMind")
        st.caption(f"Archäologische Analyse v{APP_VERSION}")
        
        # API Settings
        with st.expander("🔑 API Einstellungen", expanded=False):
            api_key = st.text_input(
                "Anthropic API Key:",
                value=st.session_state.api_key,
                type="password",
                help="Für KI-gestützte Materialerkennung"
            )
            st.session_state.api_key = api_key
            
            use_ai = st.checkbox(
                "🤖 KI-Erkennung aktivieren",
                value=bool(api_key),
                disabled=not api_key,
                help="Claude Vision für Materialanalyse"
            )
            
            if api_key:
                st.success("✓ API Key aktiv")
            else:
                st.info("💡 Ohne API: Heuristische Erkennung")
        
        st.markdown("---")
        
        # Datei-Upload
        files = st.file_uploader(
            "📤 Fundfotos hochladen", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True,
            help="Fotos von Fundstücken auf neutralem Hintergrund"
        )
        
        st.markdown("---")
        st.subheader("⚙️ Parameter")
        min_area = st.slider("Min. Fundstückgröße", 50, 1000, 200, 10)
        cluster_dist = st.slider("Cluster-Sensitivität", 0.1, 10.0, 1.5, 0.1,
                                 help="Niedriger = mehr Cluster, Höher = weniger Cluster")
        
        # Grabungskontext
        st.markdown("---")
        st.subheader("🗺️ Grabungskontext")
        excavation_name = st.text_input(
            "Grabung/Kampagne:",
            value=f"Grabung_{datetime.now().strftime('%Y')}",
            help="Name der Grabungskampagne"
        )
        layer = st.text_input(
            "Schicht/Befund:",
            placeholder="z.B. Schicht 3, Befund 42",
            help="Stratigraphische Einheit"
        )

        st.markdown("---")
        
        # Analyse starten
        if st.button("🔬 Analyse starten", type="primary", use_container_width=True):
            if not files:
                st.warning("⚠️ Bitte Fundfotos hochladen!")
            else:
                with st.spinner("Analysiere Fundstücke..."):
                    all_found = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, f in enumerate(files):
                        status_text.text(f"Bild {i+1}/{len(files)}: {f.name}")
                        
                        try:
                            img = cv2.imdecode(
                                np.asarray(bytearray(f.read()), dtype=np.uint8), 
                                cv2.IMREAD_COLOR
                            )
                            
                            def update_progress(current, total):
                                progress_bar.progress((i + current/total) / len(files))
                            
                            pieces = segment_fragments(
                                img, 
                                min_area, 
                                excavation=excavation_name,
                                api_key=st.session_state.api_key if use_ai else None,
                                use_ai=use_ai,
                                progress_callback=update_progress
                            )
                            
                            # Layer hinzufügen
                            for p in pieces:
                                p['layer'] = layer
                                p['source_file'] = f.name
                            
                            all_found.extend(pieces)
                            progress_bar.progress((i + 1) / len(files))
                        
                        except Exception as e:
                            st.error(f"Fehler bei {f.name}: {e}")
                            continue

                    # Features extrahieren
                    valid = []
                    for p in all_found:
                        feat = get_features(p)
                        if feat:
                            p['features'] = feat
                            valid.append(p)

                    st.session_state.pieces = valid
                    st.session_state.cluster_names = {}
                    st.session_state.show_tutorial = False
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✓ {len(valid)} Fundstücke aus {len(files)} Fotos erkannt!")
                    if len(valid) > 0:
                        st.info(f"🗺️ Grabung: {excavation_name}")
                    
                    st.rerun()

        if st.button("🗑️ Session leeren", use_container_width=True):
            st.session_state.pieces = []
            st.session_state.cluster_names = {}
            st.rerun()

        st.markdown("---")
        
        # Datenbank-Info
        st.subheader("💾 Datenbank")
        st.metric("Fundstücke", len(db['pieces']))
        st.metric("Gruppen", len(db.get('clusters', {})))
        
        # Hilfe
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❓ Hilfe", use_container_width=True):
                st.session_state.show_tutorial = True
                st.rerun()
        with col2:
            if st.button("📋 Changelog", use_container_width=True):
                st.session_state.show_changelog = True
                st.rerun()
        
        # PDF Export
        st.markdown("---")
        st.subheader("🖨️ Labels drucken")
        
        pdf_source = st.radio(
            "Quelle:",
            ["Aktuelle Session", "Gesamte Datenbank", "Einzelnes Fundstück"],
            help="Wähle welche Fundstücke als PDF exportiert werden"
        )
        
        if pdf_source == "Aktuelle Session" and st.session_state.pieces:
            active = [p for p in st.session_state.pieces if not p['deleted']]
            st.caption(f"{len(active)} Fundstücke")
            if st.button("📄 PDF erstellen", use_container_width=True, key="pdf_session"):
                with st.spinner("Erstelle PDF..."):
                    pdf = create_multi_label_pdf(active, st.session_state.cluster_names)
                    st.download_button(
                        "⬇️ PDF herunterladen",
                        data=pdf,
                        file_name=f"shardmind_session_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        elif pdf_source == "Gesamte Datenbank" and db['pieces']:
            st.caption(f"{len(db['pieces'])} Fundstücke")
            if st.button("📄 PDF erstellen", use_container_width=True, key="pdf_db"):
                with st.spinner("Erstelle Datenbank-PDF..."):
                    pdf = create_database_export_pdf(db)
                    st.download_button(
                        "⬇️ PDF herunterladen",
                        data=pdf,
                        file_name=f"shardmind_datenbank_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        elif pdf_source == "Einzelnes Fundstück":
            # Kombiniere Session und DB für Auswahl
            all_ids = []
            if st.session_state.pieces:
                all_ids.extend([(p['id'], f"{p['id']} (Session)") for p in st.session_state.pieces if not p.get('deleted')])
            if db['pieces']:
                all_ids.extend([(pid, f"{pid} (DB)") for pid in db['pieces'].keys()])
            
            if all_ids:
                selected_id = st.selectbox(
                    "Fundstück wählen:",
                    options=[x[0] for x in all_ids],
                    format_func=lambda x: next((y[1] for y in all_ids if y[0] == x), x)
                )
                
                if st.button("📄 Einzel-Label", use_container_width=True, key="pdf_single"):
                    # Finde das Fundstück
                    piece = None
                    for p in st.session_state.pieces:
                        if p['id'] == selected_id:
                            piece = p
                            break
                    if not piece and selected_id in db['pieces']:
                        piece = db['pieces'][selected_id]
                    
                    if piece:
                        cluster_name = None
                        if 'cluster' in piece and piece['cluster'] >= 0:
                            cluster_name = st.session_state.cluster_names.get(piece['cluster'])
                        
                        pdf = create_single_label_pdf(piece, cluster_name)
                        st.download_button(
                            "⬇️ Label herunterladen",
                            data=pdf,
                            file_name=f"label_{selected_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    # ==========================================================================
    # HAUPTBEREICH
    # ==========================================================================
    
    # Changelog anzeigen
    if st.session_state.get('show_changelog'):
        st.title("📋 Changelog")
        st.markdown(f"""
        ## Version {APP_VERSION} (2025-01-11)
        
        ### 🆕 Neue Features
        - **Eindeutige UUIDs** für alle Fundstücke (Format: SM-XXXXXXXX)
        - **Verbesserte QR-Codes** mit UUID-basierter Identifikation
        - **Flexibler PDF-Export**: Gesamte Datenbank, Session oder einzelne Fundstücke
        - **Archäologie-fokussierte UI** mit passender Terminologie
        - **KI-gestützte Materialerkennung** via Claude Vision API
        - **Automatische Klassifikation**: Keramik, Glas, Knochen, Metall, etc.
        - **Grabungskontext**: Kampagne, Schicht, Befund
        
        ### 🔧 Verbesserungen
        - Professionelle archäologische Terminologie
        - Erweiterte Metadaten pro Fundstück
        - Verbesserte Cluster-Benennung im PDF
        - Übersichtlicheres Label-Layout (3×4 pro Seite)
        
        ### 🐛 Bugfixes
        - IDs sind jetzt wirklich eindeutig (UUID statt fortlaufend)
        - Cluster-Namen werden korrekt im PDF angezeigt
        - QR-Codes enthalten vollständige Fundstück-Referenz
        """)
        
        if st.button("← Zurück"):
            st.session_state.show_changelog = False
            st.rerun()
        return
    
    # Tutorial
    if st.session_state.show_tutorial and not st.session_state.pieces:
        st.title("🏺 ShardMind - Archäologische Scherben-Analyse")
        st.markdown(f"### Version {APP_VERSION}")
        
        st.info("""
        **ShardMind** ist ein KI-gestütztes Werkzeug für die archäologische Fundbearbeitung:
        
        - 🔬 **Automatische Segmentierung** von Fundstücken aus Fotos
        - 🤖 **KI-Materialerkennung**: Keramik, Glas, Knochen, Metall, Stein...
        - 🎨 **Intelligentes Clustering**: Gruppiert zusammengehörige Scherben
        - 🔍 **Matching-Algorithmus**: Findet passende Fragmente
        - 💾 **Datenbank**: Verwaltet Funde mit Grabungskontext
        - 🖨️ **QR-Labels**: Druckbare Etiketten für physische Objekte
        """)
        
        st.warning("""
        **🔑 Für beste Ergebnisse:**
        
        Trage deinen Anthropic API Key in den Einstellungen ein. Claude Vision analysiert dann
        jedes Fundstück und erkennt Material, Objekttyp und Erhaltungszustand.
        
        **Ohne API Key:** Heuristische Erkennung basierend auf Farbe und Form.
        """)
        
        st.markdown("---")
        
        tabs = st.tabs(["📸 Aufnahme", "🔬 Analyse", "💾 Datenbank", "🖨️ Export"])
        
        with tabs[0]:
            st.markdown("### 📸 Schritt 1: Fundstücke fotografieren")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **✅ Anforderungen:**
                - Neutraler Hintergrund (weiß, grau, schwarz)
                - Gleichmäßige, schattenfreie Beleuchtung
                - Fundstücke berühren sich nicht
                - Maßstab im Bild (optional)
                - Auflösung: min. 1920×1080
                """)
            with col2:
                st.markdown("""
                **📋 Workflow:**
                1. Fundstücke auf Fototisch legen
                2. Grabungskontext notieren
                3. Mehrere Fotos pro Fundgruppe
                4. In ShardMind hochladen
                """)
        
        with tabs[1]:
            st.markdown("### 🔬 Schritt 2: Automatische Analyse")
            st.markdown("""
            **Nach dem Upload:**
            
            1. **Segmentierung**: ShardMind erkennt einzelne Fundstücke
            2. **Klassifikation**: KI bestimmt Material und Objekttyp
            3. **Benennung**: Automatisch generiert (z.B. "Keramik_Randscherbe_Rotbraun")
            4. **Clustering**: Ähnliche Fundstücke werden gruppiert
            5. **Eindeutige ID**: Jedes Fundstück erhält eine UUID (SM-XXXXXXXX)
            
            **Beispiel-Namen:**
            - `Keramik_Gefäßscherbe_Terrakotta`
            - `Glas_Flaschenfragment_Grün`
            - `Knochen_Knochenfragment_Weiß`
            - `Metall_Münze_Dunkelgrau`
            """)
        
        with tabs[2]:
            st.markdown("### 💾 Schritt 3: Datenbank verwalten")
            st.markdown("""
            **Funktionen:**
            
            - **Suchen**: Nach ID, Name, Material, Grabung
            - **Gruppieren**: Cluster benennen (z.B. "Gefäß A", "Randscherben Befund 12")
            - **Verknüpfen**: Fundstücke zu Grabungen zuordnen
            - **Exportieren**: Als PDF oder Datensicherung
            
            **Metadaten pro Fundstück:**
            - Eindeutige ID (SM-XXXXXXXX)
            - Material, Objekttyp, Farbe
            - Grabung, Schicht, Befund
            - Erstellungsdatum, Notizen
            """)
        
        with tabs[3]:
            st.markdown("### 🖨️ Schritt 4: Labels drucken")
            st.markdown("""
            **PDF-Export Optionen:**
            
            1. **Aktuelle Session**: Alle neu analysierten Fundstücke
            2. **Gesamte Datenbank**: Alle gespeicherten Fundstücke
            3. **Einzelnes Fundstück**: Ein spezifisches Label
            
            **Label enthält:**
            - QR-Code (scannbar → öffnet Fundstück in ShardMind)
            - Eindeutige ID (SM-XXXXXXXX)
            - Name (Material_Typ_Farbe)
            - Grabung & Gruppe
            
            **Anwendung:**
            - Labels ausdrucken & ausschneiden
            - Auf Fundtüte/Karton kleben
            - QR-Code scannen = sofortige Identifikation
            """)
        
        st.markdown("---")
        st.success("🚀 Bereit? Lade Fundfotos in der Sidebar hoch!")

    elif st.session_state.pieces:
        # HAUPTANSICHT MIT FUNDSTÜCKEN
        active_pieces = [p for p in st.session_state.pieces if not p['deleted']]
        
        # Clustering
        if len(active_pieces) > 1:
            labels = cluster_fragments(active_pieces, cluster_dist)
            for i, p in enumerate(active_pieces):
                p['cluster'] = labels[i]
        else:
            for p in active_pieces:
                p['cluster'] = -1
        
        cluster_ids = set([p.get('cluster', -1) for p in active_pieces])
        n_clusters = len([c for c in cluster_ids if c >= 0])
        n_single = sum(1 for p in active_pieces if p.get('cluster', -1) == -1)
        
        # Metriken
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏺 Fundstücke", len(active_pieces))
        col2.metric("📦 Gruppen", n_clusters)
        col3.metric("⚪ Einzelfunde", n_single)
        col4.metric("🗺️ Grabung", excavation_name[:15] + "..." if len(excavation_name) > 15 else excavation_name)
        
        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏺 Fundstücke", 
            "📦 Gruppen", 
            "🔍 Matching", 
            "💾 Datenbank", 
            "📷 QR-Scanner",
            "⚙️ Verwalten"
        ])

        # TAB 1: FUNDSTÜCKE (Galerie)
        with tab1:
            st.header("Erkannte Fundstücke")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Neu clustern"):
                    labels = cluster_fragments(active_pieces, cluster_dist)
                    for i, p in enumerate(active_pieces):
                        p['cluster'] = labels[i]
                    st.rerun()
            
            with col_btn2:
                if st.button("🤖 Namen neu generieren") and st.session_state.api_key:
                    with st.spinner("KI-Analyse läuft..."):
                        progress = st.progress(0)
                        for i, p in enumerate(active_pieces):
                            p['name'] = generate_fragment_name(
                                p, 
                                st.session_state.api_key, 
                                use_ai=True
                            )
                            progress.progress((i + 1) / len(active_pieces))
                        st.rerun()
            
            # Filter
            material_filter = st.multiselect(
                "Material filtern:",
                options=list(set(p.get('material', 'Unbekannt') for p in active_pieces)),
                default=[]
            )
            
            displayed = active_pieces
            if material_filter:
                displayed = [p for p in active_pieces if p.get('material') in material_filter]
            
            cols = st.columns(5)
            for i, p in enumerate(displayed):
                with cols[i % 5]:
                    cluster_id = p.get('cluster', -1)
                    color = get_cluster_color(cluster_id)
                    
                    st.markdown(
                        f'<div style="border: 3px solid {color}; padding: 3px; border-radius: 5px;">'
                        f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;">'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # ID (kurz)
                    st.caption(f"**{p['id']}**")
                    
                    # Name
                    name = p.get('name', 'Unbenannt')
                    if len(name) > 25:
                        name = name[:22] + "..."
                    st.caption(f"🏷️ {name}")
                    
                    # Material & Cluster
                    mat = p.get('material', '?')
                    cl = f"G{cluster_id}" if cluster_id >= 0 else "—"
                    st.caption(f"📦 {mat} | {cl}")
                    
                    if st.button(f"🔍", key=f"sel_{p['id']}", use_container_width=True):
                        st.session_state.selected_id = p['id']
                        st.rerun()

        # TAB 2: GRUPPEN (Cluster)
        with tab2:
            st.header("📦 Fundgruppen verwalten")
            cluster_ids_sorted = sorted([c for c in cluster_ids if c >= 0])
            
            if not cluster_ids_sorted:
                st.warning("⚠️ Keine Gruppen erkannt")
                st.info("💡 Versuche die Cluster-Sensitivität zu erhöhen (2.0-3.0)")
            else:
                for cluster_id in cluster_ids_sorted:
                    cluster_pieces = [p for p in active_pieces if p.get('cluster') == cluster_id]
                    
                    # Automatischer Gruppenname basierend auf häufigstem Material
                    materials = [p.get('material', 'Unbekannt') for p in cluster_pieces]
                    common_material = max(set(materials), key=materials.count)
                    default_name = st.session_state.cluster_names.get(
                        cluster_id, 
                        f"{common_material}_Gruppe_{cluster_id}"
                    )
                    
                    with st.expander(f"📦 Gruppe {cluster_id}: {default_name} ({len(cluster_pieces)} Funde)", expanded=True):
                        cluster_name = st.text_input(
                            "Gruppenname:", 
                            value=default_name, 
                            key=f"name_{cluster_id}",
                            help="z.B. 'Gefäß A', 'Randscherben Befund 12'"
                        )
                        st.session_state.cluster_names[cluster_id] = cluster_name
                        
                        # Vorschau
                        preview_cols = st.columns(min(6, len(cluster_pieces)))
                        for i, p in enumerate(cluster_pieces[:6]):
                            with preview_cols[i]:
                                st.image(p['thumbnail'])
                                st.caption(f"{p['id'][:11]}")
                        
                        if len(cluster_pieces) > 6:
                            st.caption(f"... +{len(cluster_pieces) - 6} weitere")
                        
                        # Aktionen
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            if st.button(
                                f"💾 Gruppe speichern", 
                                key=f"save_{cluster_id}", 
                                use_container_width=True
                            ):
                                # Zur Datenbank hinzufügen
                                cluster_key = f"{cluster_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                db['clusters'][cluster_key] = {
                                    'name': cluster_name,
                                    'created': datetime.now().isoformat(),
                                    'piece_count': len(cluster_pieces),
                                    'piece_ids': [p['id'] for p in cluster_pieces]
                                }
                                
                                for p in cluster_pieces:
                                    p['cluster_key'] = cluster_key
                                    db['pieces'][p['id']] = {
                                        'id': p['id'],
                                        'name': p.get('name'),
                                        'material': p.get('material'),
                                        'object_type': p.get('object_type'),
                                        'color_name': p.get('color_name'),
                                        'excavation': p.get('excavation'),
                                        'layer': p.get('layer'),
                                        'thumbnail': p['thumbnail'],
                                        'features': p.get('features'),
                                        'area': p['area'],
                                        'cluster_key': cluster_key,
                                        'created': p.get('created', datetime.now().isoformat())
                                    }
                                
                                save_database(db)
                                st.success(f"✓ {len(cluster_pieces)} Funde gespeichert!")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"📄 Gruppen-PDF", key=f"pdf_{cluster_id}", use_container_width=True):
                                pdf = create_multi_label_pdf(
                                    cluster_pieces,
                                    {cluster_id: cluster_name},
                                    title=f"Gruppe: {cluster_name}"
                                )
                                st.download_button(
                                    "⬇️ Download",
                                    data=pdf,
                                    file_name=f"gruppe_{cluster_id}.pdf",
                                    mime="application/pdf"
                                )
                        
                        with col3:
                            st.metric("Funde", len(cluster_pieces))

        # TAB 3: MATCHING
        with tab3:
            st.header("🔍 Passende Fragmente finden")
            
            if 'selected_id' in st.session_state:
                target = next((p for p in active_pieces if p['id'] == st.session_state.selected_id), None)
                if target:
                    st.subheader(f"Matches für {target['id']}")
                    
                    col_l, col_r = st.columns([1, 3])
                    with col_l:
                        st.markdown("**Ausgewählt:**")
                        st.image(target['thumbnail'], width=200)
                        st.write(f"**ID:** {target['id']}")
                        st.write(f"**Name:** {target.get('name', 'N/A')}")
                        st.write(f"**Material:** {target.get('material', 'N/A')}")
                    
                    with col_r:
                        st.markdown("**Top 10 Matches:**")
                        
                        # Berechne Matches
                        matches = []
                        for p in active_pieces:
                            if p['id'] != target['id'] and 'features' in p and 'features' in target:
                                score = calculate_match_score(target['features'], p['features'])
                                matches.append((score, p))
                        
                        matches.sort(key=lambda x: x[0], reverse=True)
                        
                        for row in range(2):
                            m_cols = st.columns(5)
                            for col in range(5):
                                idx = row * 5 + col
                                if idx >= len(matches):
                                    break
                                score, p = matches[idx]
                                with m_cols[col]:
                                    st.image(p['thumbnail'])
                                    st.caption(f"{p['id'][:11]}")
                                    st.progress(score / 100)
                                    
                                    # Farbcodierung
                                    if score > 80:
                                        st.markdown(f"**:green[{score:.1f}%]**")
                                    elif score > 60:
                                        st.markdown(f"**:orange[{score:.1f}%]**")
                                    else:
                                        st.markdown(f"**:red[{score:.1f}%]**")
            else:
                st.info("👈 Wähle ein Fundstück in der Galerie aus")
                st.markdown("""
                **So funktioniert's:**
                1. Gehe zu "🏺 Fundstücke"
                2. Klicke auf 🔍 bei einem Fundstück
                3. ShardMind zeigt die ähnlichsten Fragmente
                
                **Interpretation:**
                - 🟢 >80%: Sehr wahrscheinlich zusammengehörig
                - 🟠 60-80%: Möglicherweise verwandt
                - 🔴 <60%: Wahrscheinlich unterschiedlich
                """)

        # TAB 4: DATENBANK
        with tab4:
            st.header("💾 Funddatenbank")
            
            db_tab1, db_tab2, db_tab3 = st.tabs(["📋 Alle Funde", "📦 Gruppen", "⚙️ Verwalten"])
            
            with db_tab1:
                if not db['pieces']:
                    st.info("Datenbank ist leer. Speichere Fundstücke über den Gruppen-Tab.")
                else:
                    # Suchfilter
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        search_id = st.text_input("🔍 ID suchen:", placeholder="SM-XXXXXXXX")
                    with col2:
                        search_material = st.selectbox(
                            "Material:",
                            ["Alle"] + list(set(p.get('material', '') for p in db['pieces'].values() if p.get('material')))
                        )
                    with col3:
                        search_excavation = st.text_input("Grabung:", placeholder="Grabungsname")
                    
                    # Filtern
                    filtered = list(db['pieces'].values())
                    
                    if search_id:
                        filtered = [p for p in filtered if search_id.upper() in p['id'].upper()]
                    if search_material != "Alle":
                        filtered = [p for p in filtered if p.get('material') == search_material]
                    if search_excavation:
                        filtered = [p for p in filtered if search_excavation.lower() in p.get('excavation', '').lower()]
                    
                    st.caption(f"Zeige {len(filtered)} von {len(db['pieces'])} Fundstücken")
                    
                    if filtered:
                        # Pagination
                        per_page = st.selectbox("Pro Seite:", [12, 24, 48], index=0)
                        total_pages = max(1, (len(filtered) - 1) // per_page + 1)
                        page = st.slider("Seite", 1, total_pages, 1) if total_pages > 1 else 1
                        
                        start = (page - 1) * per_page
                        end = min(start + per_page, len(filtered))
                        
                        for row in range(0, end - start, 6):
                            cols = st.columns(6)
                            for col_idx in range(6):
                                idx = start + row + col_idx
                                if idx >= end:
                                    break
                                p = filtered[idx]
                                with cols[col_idx]:
                                    if 'thumbnail' in p:
                                        st.image(p['thumbnail'])
                                    st.caption(f"**{p['id']}**")
                                    st.caption(f"🏷️ {p.get('name', 'N/A')[:20]}")
                                    st.caption(f"📦 {p.get('material', '?')}")
            
            with db_tab2:
                if not db.get('clusters'):
                    st.info("Keine Gruppen gespeichert.")
                else:
                    for cluster_key, cluster_data in db['clusters'].items():
                        with st.expander(f"📦 {cluster_data['name']} ({cluster_data['piece_count']} Funde)"):
                            st.write(f"**Erstellt:** {cluster_data['created'][:10]}")
                            
                            # Zeige Fundstücke der Gruppe
                            piece_ids = cluster_data.get('piece_ids', [])
                            preview_pieces = [db['pieces'][pid] for pid in piece_ids[:6] if pid in db['pieces']]
                            
                            if preview_pieces:
                                cols = st.columns(6)
                                for i, p in enumerate(preview_pieces):
                                    with cols[i]:
                                        if 'thumbnail' in p:
                                            st.image(p['thumbnail'])
                                        st.caption(p['id'][:11])
            
            with db_tab3:
                st.subheader("⚙️ Datenbank verwalten")
                
                # Einzelne Fundstücke hinzufügen
                with st.expander("➕ Fundstücke hinzufügen", expanded=False):
                    if not st.session_state.pieces:
                        st.info("Erst Fundfotos analysieren!")
                    else:
                        active_current = [p for p in st.session_state.pieces if not p['deleted']]
                        
                        st.markdown(f"**{len(active_current)} Fundstücke in aktueller Session**")
                        
                        selected_ids = st.multiselect(
                            "Fundstücke auswählen:",
                            options=[p['id'] for p in active_current],
                            format_func=lambda x: f"{x} - {next((p.get('name', 'N/A')[:30] for p in active_current if p['id'] == x), 'N/A')}"
                        )
                        
                        if selected_ids:
                            if st.button(f"➕ {len(selected_ids)} Fundstücke hinzufügen", type="primary"):
                                added = 0
                                for pid in selected_ids:
                                    piece = next((p for p in active_current if p['id'] == pid), None)
                                    if piece and piece['id'] not in db['pieces']:
                                        db['pieces'][piece['id']] = {
                                            'id': piece['id'],
                                            'name': piece.get('name'),
                                            'material': piece.get('material'),
                                            'object_type': piece.get('object_type'),
                                            'color_name': piece.get('color_name'),
                                            'excavation': piece.get('excavation'),
                                            'layer': piece.get('layer'),
                                            'thumbnail': piece['thumbnail'],
                                            'features': piece.get('features'),
                                            'area': piece['area'],
                                            'created': piece.get('created', datetime.now().isoformat())
                                        }
                                        added += 1
                                
                                save_database(db)
                                st.success(f"✓ {added} Fundstücke hinzugefügt!")
                                st.rerun()
                
                st.markdown("---")
                
                # Löschen
                with st.expander("🗑️ Daten löschen", expanded=False):
                    st.warning("⚠️ Diese Aktionen sind permanent!")
                    
                    if db['pieces']:
                        delete_id = st.selectbox(
                            "Fundstück löschen:",
                            options=[""] + list(db['pieces'].keys()),
                            format_func=lambda x: f"{x} - {db['pieces'][x].get('name', 'N/A')[:30]}" if x else "Auswählen..."
                        )
                        
                        if delete_id and st.button(f"🗑️ {delete_id} löschen"):
                            del db['pieces'][delete_id]
                            save_database(db)
                            st.success("Gelöscht!")
                            st.rerun()
                    
                    st.markdown("---")
                    
                    confirm = st.text_input("Zum Löschen der gesamten DB 'LÖSCHEN' eingeben:")
                    if st.button("🔥 GESAMTE DATENBANK LÖSCHEN", type="secondary"):
                        if confirm == "LÖSCHEN":
                            save_database(create_empty_database())
                            st.success("Datenbank gelöscht!")
                            st.rerun()
                        else:
                            st.error("Bestätigung erforderlich!")

        # TAB 5: QR-SCANNER
        with tab5:
            st.header("📷 QR-Code Scanner")
            st.markdown("Scanne QR-Codes von Labels um Fundstücke zu identifizieren")
            
            qr_input = st.text_input(
                "QR-Code / ID eingeben:",
                placeholder="shardmind://find/SM-XXXXXXXX oder SM-XXXXXXXX",
                help="Scanne den QR-Code oder gib die ID manuell ein"
            )
            
            if qr_input:
                # Parse Input
                piece_id = None
                
                if "shardmind://find/" in qr_input:
                    # QR-Code Format
                    parts = qr_input.replace("shardmind://find/", "").split("?")
                    piece_id = parts[0]
                elif qr_input.startswith("SM-"):
                    # Direkte ID
                    piece_id = qr_input
                else:
                    st.error("Ungültiges Format. Erwartet: SM-XXXXXXXX")
                
                if piece_id:
                    # Suche in Session
                    session_piece = next(
                        (p for p in st.session_state.pieces if p['id'] == piece_id and not p.get('deleted')), 
                        None
                    )
                    
                    # Suche in Datenbank
                    db_piece = db['pieces'].get(piece_id)
                    
                    if session_piece or db_piece:
                        st.success(f"✓ Fundstück gefunden: {piece_id}")
                        
                        col1, col2 = st.columns(2)
                        
                        piece = session_piece or db_piece
                        
                        with col1:
                            st.markdown("### 🏺 Fundstück-Details")
                            if 'thumbnail' in piece:
                                st.image(piece['thumbnail'], width=250)
                            
                            st.write(f"**ID:** `{piece['id']}`")
                            st.write(f"**Name:** {piece.get('name', 'N/A')}")
                            st.write(f"**Material:** {piece.get('material', 'N/A')}")
                            st.write(f"**Objekttyp:** {piece.get('object_type', 'N/A')}")
                            st.write(f"**Farbe:** {piece.get('color_name', 'N/A')}")
                        
                        with col2:
                            st.markdown("### 🗺️ Kontext")
                            st.write(f"**Grabung:** {piece.get('excavation', 'N/A')}")
                            st.write(f"**Schicht:** {piece.get('layer', 'N/A')}")
                            st.write(f"**Erstellt:** {piece.get('created', 'N/A')[:10] if piece.get('created') else 'N/A'}")
                            
                            if piece.get('cluster_key') and piece['cluster_key'] in db.get('clusters', {}):
                                cluster = db['clusters'][piece['cluster_key']]
                                st.write(f"**Gruppe:** {cluster['name']}")
                            
                            # Aktionen
                            st.markdown("---")
                            if session_piece and st.button("🔍 Matches zeigen"):
                                st.session_state.selected_id = piece_id
                                st.rerun()
                            
                            if st.button("📄 Einzel-Label"):
                                pdf = create_single_label_pdf(piece)
                                st.download_button(
                                    "⬇️ Download",
                                    data=pdf,
                                    file_name=f"label_{piece_id}.pdf",
                                    mime="application/pdf"
                                )
                    else:
                        st.warning(f"❌ Fundstück {piece_id} nicht gefunden")
                        st.info("Das Fundstück ist weder in der aktuellen Session noch in der Datenbank.")

        # TAB 6: VERWALTEN
        with tab6:
            st.header("⚙️ Session verwalten")
            st.markdown("Fundstücke aus der aktuellen Session löschen oder wiederherstellen")
            
            all_pieces = st.session_state.pieces
            active = [p for p in all_pieces if not p['deleted']]
            deleted = [p for p in all_pieces if p['deleted']]
            
            st.write(f"**Aktiv:** {len(active)} | **Gelöscht:** {len(deleted)}")
            
            cols = st.columns(6)
            for i, p in enumerate(all_pieces):
                with cols[i % 6]:
                    if p['deleted']:
                        st.markdown(
                            f'<div style="opacity: 0.3; border: 2px solid red; padding: 3px;">'
                            f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;">'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        st.caption(f"~~{p['id'][:11]}~~")
                        if st.button(f"↩️", key=f"rev_{p['id']}", use_container_width=True):
                            p['deleted'] = False
                            st.rerun()
                    else:
                        st.image(p['thumbnail'])
                        st.caption(f"**{p['id'][:11]}**")
                        st.caption(f"{p.get('name', 'N/A')[:20]}")
                        if st.button(f"❌", key=f"del_{p['id']}", use_container_width=True):
                            p['deleted'] = True
                            st.rerun()
    
    else:
        st.info("👈 Lade Fundfotos in der Sidebar hoch, um zu beginnen")
        
        # Quick-Start
        st.markdown("---")
        st.markdown("### 🚀 Schnellstart")
        st.markdown("""
        1. **API Key eingeben** (optional, für KI-Erkennung)
        2. **Fundfotos hochladen** (neutraler Hintergrund)
        3. **Grabungskontext** eingeben
        4. **"🔬 Analyse starten"** klicken
        """)


if __name__ == "__main__":
    main()
