import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import pickle
import qrcode
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import requests
import json
import os

DB_PATH = Path("puzzle_database_v4.pkl")
FEATURE_VERSION = 4

# =============================================================================
# KI-OBJEKTERKENNUNG MIT CLAUDE API
# =============================================================================

def analyze_image_with_ai(image_base64, api_key=None):
    """
    Nutzt Claude API um das Objekt im Bild zu erkennen.
    Gibt zurück: {"object": "Zauberwürfel", "color": "Bunt", "confidence": 0.95}
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
                "max_tokens": 150,
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
                                "text": """Analysiere dieses Bild eines einzelnen Objekts.
Antworte NUR mit einem JSON-Objekt in diesem Format (keine anderen Texte):
{"object": "Objektname", "color": "Hauptfarbe", "material": "Material"}

Beispiele für Objektnamen:
- Puzzlestück, Zauberwürfel, Münze, Schlüssel, Schraube, Knopf, Spielstein
- Scherbe, Glasstück, Keramikfragment, Holzstück
- Legoteil, Dominostein, Würfel, Spielfigur

Sei präzise und spezifisch. Wenn es ein Puzzlestück ist, sage "Puzzlestück".
Wenn du es nicht erkennst, sage "Unbekanntes_Objekt"."""
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
            # Parse JSON aus Antwort
            try:
                # Bereinige mögliche Markdown-Formatierung
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
    """Extrahiert die dominante Farbe und gibt einen deutschen Namen zurück"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
    # Farbname basierend auf HSV
    if s < 30:
        if v < 80:
            return "Dunkel"
        elif v < 160:
            return "Grau"
        else:
            return "Weiß"
    elif h < 10 or h > 170:
        return "Rot"
    elif h < 25:
        return "Orange"
    elif h < 40:
        return "Gelb"
    elif h < 75:
        return "Grün"
    elif h < 130:
        return "Blau"
    elif h < 150:
        return "Cyan"
    else:
        return "Violett"


def generate_ai_name_with_fallback(piece, api_key=None, use_ai=True):
    """
    Generiert Namen mit KI-Objekterkennung (falls API verfügbar)
    Fallback auf heuristische Analyse
    
    Format: Farbe_ObjektTyp_ID (z.B. "Rot_Zauberwürfel_001")
    """
    color = get_dominant_color_name(piece['thumbnail'], piece['mask'])
    object_type = "Objekt"  # Fallback
    
    # Versuche KI-Erkennung
    if use_ai and api_key:
        img_base64 = image_to_base64_raw(piece['thumbnail'])
        ai_result = analyze_image_with_ai(img_base64, api_key)
        
        if ai_result and 'object' in ai_result:
            object_type = ai_result['object'].replace(" ", "_")
            # Nutze auch KI-Farbe wenn vorhanden
            if 'color' in ai_result and ai_result['color']:
                color = ai_result['color']
    else:
        # Fallback: Heuristische Klassifikation
        object_type = classify_object_heuristic(piece)
    
    # Format: Farbe_ObjektTyp_ID
    return f"{color}_{object_type}_{piece['id']:03d}"


def classify_object_heuristic(piece):
    """
    Heuristische Objektklassifikation basierend auf Form-Eigenschaften
    (Fallback wenn keine API verfügbar)
    """
    area = piece['area']
    perimeter = cv2.arcLength(piece['contour'], True)
    compactness = (perimeter ** 2) / (area + 1e-6)
    
    # Ecken zählen
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(piece['contour'], epsilon, True)
    corners = len(approx)
    
    # Aspect Ratio
    x, y, w, h = cv2.boundingRect(piece['contour'])
    aspect_ratio = w / (h + 1e-6)
    
    # Konvexität
    hull = cv2.convexHull(piece['contour'])
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    
    # Klassifikation
    if corners == 4 and 0.85 < aspect_ratio < 1.15 and compactness < 18:
        if solidity > 0.95:
            return "Würfel"
        else:
            return "Karte"
    
    elif corners == 3:
        return "Dreieck"
    
    elif compactness < 14 and solidity > 0.9:
        return "Münze"
    
    elif 0.4 < solidity < 0.85 and 15 < compactness < 50:
        return "Puzzlestück"
    
    elif solidity < 0.6:
        return "Scherbe"
    
    elif aspect_ratio > 3 or aspect_ratio < 0.33:
        return "Stab"
    
    elif compactness > 50:
        return "Fragment"
    
    else:
        return "Teil"


def image_to_base64_raw(img):
    """Konvertiert OpenCV-Bild zu Base64 (ohne data:image Präfix)"""
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
                    return {'pieces': [], 'clusters': {}, 'version': FEATURE_VERSION}
                return db
        except:
            return {'pieces': [], 'clusters': {}, 'version': FEATURE_VERSION}
    return {'pieces': [], 'clusters': {}, 'version': FEATURE_VERSION}


def save_database(db):
    db['version'] = FEATURE_VERSION
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)


# =============================================================================
# QR-CODE & PDF GENERATION
# =============================================================================

def generate_qr_code(data):
    """Generiert QR-Code als PIL Image"""
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def create_label_pdf(pieces, cluster_names_map, filename="puzzle_labels.pdf"):
    """
    Erstellt PDF mit QR-Codes und Labels - 3D-Drucker-optimiert
    
    Args:
        pieces: Liste der Teile
        cluster_names_map: Dict {cluster_id: cluster_name} für richtige Cluster-Namen
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Layout: 3 Spalten, 7 Zeilen pro Seite
    cols, rows = 3, 7
    cell_width = width / cols
    cell_height = height / rows
    
    x_offset, y_offset = 20, 20
    qr_size = 80
    
    active_pieces = [p for p in pieces if not p.get('deleted', False)]
    
    for idx, piece in enumerate(active_pieces):
        # Position berechnen
        col = idx % cols
        row = (idx // cols) % rows
        
        if idx > 0 and idx % (cols * rows) == 0:
            c.showPage()  # Neue Seite
        
        x = col * cell_width + x_offset
        y = height - (row + 1) * cell_height + y_offset
        
        # QR-Code Data mit Batch-Info
        qr_data = f"shardmind://piece/{piece['id']}"
        if 'batch' in piece:
            qr_data += f"?batch={piece['batch']}"
        
        qr_img = generate_qr_code(qr_data)
        
        # QR-Code in PDF einfügen
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        
        c.drawImage(ImageReader(qr_buffer), x, y, width=qr_size, height=qr_size)
        
        # Text - optimiert für 3D-Druck-Labels
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x + qr_size + 10, y + qr_size - 15, f"ID: {piece['id']}")
        
        # AI-Name (z.B. "Rot_Zauberwürfel_001")
        c.setFont("Helvetica", 9)
        ai_name = piece.get('ai_name', 'Unbenannt')
        # Kürzen falls zu lang
        if len(ai_name) > 20:
            ai_name = ai_name[:17] + "..."
        c.drawString(x + qr_size + 10, y + qr_size - 30, ai_name)
        
        y_pos = y + qr_size - 45
        
        # Batch-Info
        if 'batch' in piece:
            c.setFont("Helvetica", 8)
            batch_name = piece['batch'][:15] if len(piece['batch']) > 15 else piece['batch']
            c.drawString(x + qr_size + 10, y_pos, f"Batch: {batch_name}")
            y_pos -= 12
        
        # CLUSTER-NAME (aus cluster_names_map!)
        cluster_id = piece.get('cluster', -1)
        if cluster_id >= 0:
            # Hole den richtigen Cluster-Namen aus der Map
            cluster_name = cluster_names_map.get(cluster_id, f"Gruppe_{cluster_id}")
            c.setFont("Helvetica", 8)
            # Kürzen falls zu lang
            if len(cluster_name) > 15:
                cluster_name = cluster_name[:12] + "..."
            c.drawString(x + qr_size + 10, y_pos, f"Gruppe: {cluster_name}")
        
        # Rahmen für 3D-Druck (Schnittlinie)
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.setLineWidth(0.5)
        c.setDash(2, 2)  # Gestrichelt
        c.rect(x - 5, y - 5, cell_width - 10, cell_height - 10)
    
    c.save()
    buffer.seek(0)
    return buffer


# =============================================================================
# BILDVERARBEITUNG & SEGMENTIERUNG
# =============================================================================

def is_valid_puzzle_piece(contour, roi, mask_roi, image_shape):
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


def segment_pieces_robust(image, min_area=100, start_id=0, api_key=None, use_ai=True, progress_callback=None):
    """
    Segmentiert Puzzleteile aus einem Bild
    
    Args:
        image: OpenCV-Bild
        min_area: Minimale Fläche für Teile
        start_id: Start-ID für Teile
        api_key: Anthropic API Key für KI-Erkennung
        use_ai: KI-Erkennung nutzen?
        progress_callback: Callback für Fortschrittsanzeige
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

    pieces = []
    valid_contours = []
    
    # Erst alle validen Konturen sammeln
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
        
        if not is_valid_puzzle_piece(c, roi, mask_roi, padded.shape):
            continue
        
        valid_contours.append({
            'contour': c,
            'roi': roi,
            'mask_roi': mask_roi,
            'area': cv2.contourArea(c)
        })
    
    # Dann mit KI benennen (mit Fortschrittsanzeige)
    for i, vc in enumerate(valid_contours):
        piece = {
            'id': start_id + len(pieces),
            'contour': vc['contour'],
            'thumbnail': vc['roi'],
            'mask': vc['mask_roi'],
            'area': vc['area'],
            'deleted': False
        }
        
        # KI-Name generieren
        piece['ai_name'] = generate_ai_name_with_fallback(piece, api_key, use_ai)
        pieces.append(piece)
        
        if progress_callback:
            progress_callback(i + 1, len(valid_contours))
    
    return pieces


# =============================================================================
# FEATURE-EXTRAKTION & CLUSTERING
# =============================================================================

def get_features(p):
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


def calculate_score(f1, f2):
    dist_c = np.linalg.norm(f1['color'] - f2['color'])
    score_c = max(0, 100 - (dist_c / 3.0))
    
    s1, s2 = f1['shape'], f2['shape']
    best_diff = min([np.mean((s1 - np.roll(s2, r)) ** 2) for r in range(0, 180, 3)])
    
    score_s = 95 + (0.02 - best_diff) * 250 if best_diff < 0.02 else max(0, 95 * (1 - best_diff * 6))
    
    return min(100, max(0, (0.4 * score_c) + (0.6 * score_s)))


def cluster_pieces_smart(active_pieces, distance_threshold=1.5):
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
    if cluster_id == -1:
        return "rgb(180, 180, 180)"
    hue = int((cluster_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


def save_cluster_to_db(pieces, cluster_id, cluster_name, db):
    cluster_pieces = [p for p in pieces if p.get('cluster') == cluster_id]
    if not cluster_pieces:
        return 0
    
    cluster_key = f"{cluster_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db['clusters'][cluster_key] = {
        'name': cluster_name,
        'created': datetime.now().isoformat(),
        'cluster_id': cluster_id,
        'piece_count': len(cluster_pieces),
        'pieces': []
    }
    
    for p in cluster_pieces:
        piece_data = {
            'features': p['features'], 
            'thumbnail': p['thumbnail'], 
            'original_id': p['id'], 
            'area': p['area'],
            'ai_name': p.get('ai_name', 'Unbenannt'),
            'cluster_key': cluster_key
        }
        db['clusters'][cluster_key]['pieces'].append(piece_data)
        db['pieces'].append(piece_data)
    
    save_database(db)
    return len(cluster_pieces)


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="ShardMind - KI Scherben-Analyse", 
        page_icon="🧠",
        layout="wide"
    )

    # Session State initialisieren
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
        st.title("🧠 ShardMind")
        st.caption("KI-gestützte Scherben-Analyse")
        
        # API Key Eingabe
        with st.expander("🔑 API Einstellungen", expanded=False):
            api_key = st.text_input(
                "Anthropic API Key:",
                value=st.session_state.api_key,
                type="password",
                help="Für echte KI-Objekterkennung"
            )
            st.session_state.api_key = api_key
            
            use_ai = st.checkbox(
                "🤖 KI-Erkennung nutzen",
                value=bool(api_key),
                disabled=not api_key,
                help="Aktiviert Claude Vision für Objekterkennung"
            )
            
            if api_key:
                st.success("✓ API Key gesetzt")
            else:
                st.info("💡 Ohne API Key: Heuristische Erkennung")
        
        st.markdown("---")
        
        files = st.file_uploader(
            "📤 Bilder hochladen", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
        
        st.markdown("---")
        st.subheader("🔧 Parameter")
        min_area = st.slider("Min. Teilgröße", 50, 1000, 200, 10)
        cluster_dist = st.slider("Cluster-Distanz", 0.1, 10.0, 1.5, 0.1)
        
        # Batch-Name
        batch_name = st.text_input(
            "📦 Batch-Name:",
            value=f"Batch_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="batch_name_input"
        )

        st.markdown("---")
        
        if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
            if not files:
                st.warning("⚠️ Bitte Bilder hochladen!")
            else:
                with st.spinner("Verarbeite Bilder..."):
                    all_found = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    current_id = 0
                    
                    for i, f in enumerate(files):
                        status_text.text(f"Bild {i+1}/{len(files)}: {f.name}")
                        
                        try:
                            img = cv2.imdecode(
                                np.asarray(bytearray(f.read()), dtype=np.uint8), 
                                cv2.IMREAD_COLOR
                            )
                            
                            def update_progress(current, total):
                                progress_bar.progress(
                                    (i + current/total) / len(files)
                                )
                            
                            pieces = segment_pieces_robust(
                                img, 
                                min_area, 
                                start_id=current_id,
                                api_key=st.session_state.api_key if use_ai else None,
                                use_ai=use_ai,
                                progress_callback=update_progress
                            )
                            
                            # Batch-Name zu allen Teilen hinzufügen
                            for p in pieces:
                                p['batch'] = batch_name
                            
                            all_found.extend(pieces)
                            
                            if pieces:
                                current_id = max(p['id'] for p in pieces) + 1
                            
                            progress_bar.progress((i + 1) / len(files))
                        
                        except Exception as e:
                            st.error(f"Fehler bei Bild {i+1}: {e}")
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
                    
                    st.success(f"✓ {len(valid)} Teile aus {len(files)} Bildern erkannt!")
                    if len(valid) > 0:
                        st.info(f"📦 Batch: {batch_name}")
                        if use_ai and st.session_state.api_key:
                            st.info("🤖 KI-Erkennung aktiv")
                    
                    st.rerun()

        if st.button("🗑️ Alles löschen", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("💾 Datenbank")
        st.metric("Teile", len(db['pieces']))
        st.metric("Cluster", len(db.get('clusters', {})))
        
        if st.button("❓ Tutorial", use_container_width=True):
            st.session_state.show_tutorial = True
            st.rerun()
        
        # PDF Download
        st.markdown("---")
        st.subheader("🖨️ Drucken")
        
        if st.session_state.pieces:
            active_for_print = [p for p in st.session_state.pieces if not p['deleted']]
            if active_for_print:
                st.caption(f"{len(active_for_print)} Teile bereit")
                if st.button("📄 PDF erstellen", use_container_width=True):
                    with st.spinner("Erstelle PDF..."):
                        # Übergebe cluster_names an PDF-Funktion
                        pdf_buffer = create_label_pdf(
                            active_for_print, 
                            st.session_state.cluster_names
                        )
                        st.download_button(
                            "⬇️ PDF herunterladen",
                            data=pdf_buffer,
                            file_name=f"shardmind_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            else:
                st.caption("Keine Teile zum Drucken")
        else:
            st.caption("Erst Teile analysieren")

    # ==========================================================================
    # HAUPTBEREICH
    # ==========================================================================
    
    if st.session_state.show_tutorial and not st.session_state.pieces:
        # TUTORIAL
        st.title("🧠 ShardMind - Tutorial")
        st.markdown("### KI-gestützte Scherben-Analyse")
        
        st.info("""
        **ShardMind** nutzt Computer Vision & KI zur Analyse von Puzzle-Teilen und archäologischen Scherben:
        - 🤖 **KI-Objekterkennung**: Claude Vision erkennt was das Objekt ist (Zauberwürfel, Puzzlestück, Münze...)
        - 🎨 **Auto-Clustering**: Gruppiert ähnliche Teile
        - 🔍 **Smart Matching**: Findet passende Teile
        - 💾 **Datenbank**: Speichert & verwaltet Teile mit Clustern
        - 🖨️ **QR-Code-PDF**: Drucke Labels für physische Objekte
        """)
        
        st.warning("""
        **🔑 Für beste Ergebnisse:**
        - Trage deinen Anthropic API Key in den Einstellungen ein
        - Aktiviere "KI-Erkennung nutzen"
        - Claude Vision analysiert dann jedes Objekt und benennt es automatisch
        
        **Ohne API Key:** Heuristische Erkennung basierend auf Form/Farbe
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📸 Beispiel-Namen MIT KI:")
            st.code("""
Rot_Zauberwürfel_001
Grau_Puzzlestück_002
Silber_Münze_003
Braun_Holzscherbe_004
Blau_Legoteil_005
            """)
        
        with col2:
            st.markdown("### 📸 Beispiel-Namen OHNE KI:")
            st.code("""
Rot_Würfel_001
Grau_Puzzlestück_002
Grau_Münze_003
Braun_Scherbe_004
Blau_Teil_005
            """)
        
        st.success("🚀 Bereit? Lade Bilder hoch!")

    elif st.session_state.pieces:
        # HAUPTANSICHT MIT TEILEN
        active_pieces = [p for p in st.session_state.pieces if not p['deleted']]
        
        # Clustering
        if len(active_pieces) > 1:
            labels = cluster_pieces_smart(active_pieces, cluster_dist)
            for i, p in enumerate(active_pieces):
                p['cluster'] = labels[i]
        else:
            for p in active_pieces:
                p['cluster'] = -1
        
        cluster_ids = set([p.get('cluster', -1) for p in active_pieces])
        n_clusters = len([c for c in cluster_ids if c >= 0])
        n_noise = sum(1 for p in active_pieces if p.get('cluster', -1) == -1)
        
        # Metriken
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Teile", len(active_pieces))
        col2.metric("🎯 Cluster", n_clusters)
        col3.metric("⚪ Einzeln", n_noise)
        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Galerie", 
            "📦 Cluster-Manager", 
            "🔍 Matching", 
            "💾 Datenbank", 
            "📷 QR-Scanner",
            "❌ Verwalten"
        ])

        # TAB 1: GALERIE
        with tab1:
            st.header("Erkannte Teile")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Clustering neu berechnen"):
                    labels = cluster_pieces_smart(active_pieces, cluster_dist)
                    for i, p in enumerate(active_pieces):
                        p['cluster'] = labels[i]
                    st.rerun()
            
            with col_btn2:
                if st.button("🤖 Namen neu generieren") and st.session_state.api_key:
                    with st.spinner("Generiere KI-Namen..."):
                        progress = st.progress(0)
                        for i, p in enumerate(active_pieces):
                            p['ai_name'] = generate_ai_name_with_fallback(
                                p, 
                                st.session_state.api_key, 
                                use_ai=True
                            )
                            progress.progress((i + 1) / len(active_pieces))
                        st.rerun()
            
            cols = st.columns(6)
            for i, p in enumerate(active_pieces):
                with cols[i % 6]:
                    cluster_id = p.get('cluster', -1)
                    color = get_cluster_color(cluster_id)
                    cluster_label = "Einzeln" if cluster_id == -1 else f"C{cluster_id}"
                    
                    st.markdown(
                        f'<div style="border: 3px solid {color}; padding: 3px;">'
                        f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;">'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"ID: {p['id']} | {cluster_label}")
                    st.caption(f"🤖 {p.get('ai_name', 'N/A')}")
                    if st.button(f"🔍", key=f"sel_{p['id']}", use_container_width=True):
                        st.session_state.selected_id = p['id']
                        st.rerun()

        # TAB 2: CLUSTER-MANAGER
        with tab2:
            st.header("📦 Cluster verwalten")
            cluster_ids = sorted([c for c in cluster_ids if c >= 0])
            
            if not cluster_ids:
                st.warning("⚠️ Keine Cluster!")
                st.info("💡 Erhöhe Cluster-Distanz (2.0-3.0)")
            else:
                for cluster_id in cluster_ids:
                    cluster_pieces = [p for p in active_pieces if p.get('cluster') == cluster_id]
                    with st.expander(f"🎯 Cluster {cluster_id} ({len(cluster_pieces)} Teile)", expanded=True):
                        default_name = st.session_state.cluster_names.get(cluster_id, f"Cluster_{cluster_id}")
                        cluster_name = st.text_input(
                            "Name:", 
                            value=default_name, 
                            key=f"name_{cluster_id}"
                        )
                        st.session_state.cluster_names[cluster_id] = cluster_name
                        
                        preview_cols = st.columns(min(6, len(cluster_pieces)))
                        for i, p in enumerate(cluster_pieces[:6]):
                            with preview_cols[i]:
                                st.image(p['thumbnail'])
                                st.caption(f"{p.get('ai_name', 'N/A')}")
                        
                        if len(cluster_pieces) > 6:
                            st.caption(f"... +{len(cluster_pieces) - 6}")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button(
                                f"💾 '{cluster_name}' speichern", 
                                key=f"save_{cluster_id}", 
                                use_container_width=True
                            ):
                                count = save_cluster_to_db(active_pieces, cluster_id, cluster_name, db)
                                st.success(f"✓ {count} Teile!")
                                st.rerun()
                        with col2:
                            st.metric("Teile", len(cluster_pieces))

        # TAB 3: MATCHING
        with tab3:
            if 'selected_id' in st.session_state:
                target = next((p for p in active_pieces if p['id'] == st.session_state.selected_id), None)
                if target:
                    st.header(f"Matches für #{target['id']} ({target.get('ai_name', 'N/A')})")
                    col_l, col_r = st.columns([1, 4])
                    with col_l:
                        st.markdown("**Ausgewählt:**")
                        st.image(target['thumbnail'], width=200)
                        st.caption(f"🤖 {target.get('ai_name', 'N/A')}")
                    with col_r:
                        st.markdown("**Top 10:**")
                        matches = sorted(
                            [(calculate_score(target['features'], p['features']), p) 
                             for p in active_pieces if p['id'] != target['id']], 
                            key=lambda x: x[0], 
                            reverse=True
                        )
                        for row in range(2):
                            m_cols = st.columns(5)
                            for col in range(5):
                                idx = row * 5 + col
                                if idx >= len(matches):
                                    break
                                score, p = matches[idx]
                                with m_cols[col]:
                                    st.image(p['thumbnail'])
                                    st.caption(f"{p.get('ai_name', 'N/A')}")
                                    st.progress(score / 100)
                                    st.markdown(f"**{score:.1f}%**")
            else:
                st.info("👈 Teil wählen")

        # TAB 4: DATENBANK
        with tab4:
            st.header("💾 Datenbank")
            db_tab1, db_tab2, db_tab3 = st.tabs(["📋 Alle Teile", "📦 Cluster", "🗑️ Editor"])
            
            with db_tab1:
                st.subheader("📋 Alle Teile")
                
                # Suchfilter
                col1, col2, col3 = st.columns(3)
                with col1:
                    search_name = st.text_input("🔍 Nach Name suchen:", key="search_name")
                with col2:
                    search_batch = st.text_input("📦 Nach Batch suchen:", key="search_batch")
                with col3:
                    search_id = st.text_input("🔢 Nach ID suchen:", key="search_id")
                
                if not db['pieces']:
                    st.info("DB leer")
                else:
                    filtered_pieces = db['pieces']
                    
                    if search_name:
                        filtered_pieces = [
                            p for p in filtered_pieces 
                            if search_name.lower() in p.get('ai_name', '').lower()
                        ]
                    
                    if search_batch:
                        filtered_pieces = [
                            p for p in filtered_pieces 
                            if search_batch.lower() in str(p.get('batch', '')).lower()
                        ]
                    
                    if search_id:
                        try:
                            search_id_int = int(search_id)
                            filtered_pieces = [
                                p for p in filtered_pieces 
                                if p.get('original_id') == search_id_int
                            ]
                        except:
                            pass
                    
                    st.caption(f"Zeige {len(filtered_pieces)} von {len(db['pieces'])} Teilen")
                    
                    if not filtered_pieces:
                        st.warning("Keine Teile gefunden")
                    else:
                        show_per_page = st.selectbox("Pro Seite", [12, 24, 48], index=1)
                        total_pages = max(1, (len(filtered_pieces) - 1) // show_per_page + 1)
                        page = st.slider("Seite", 1, total_pages, 1, key="db_page_slider") if total_pages > 1 else 1
                        start_idx = (page - 1) * show_per_page
                        end_idx = min(start_idx + show_per_page, len(filtered_pieces))
                        
                        for row in range(0, end_idx - start_idx, 6):
                            cols = st.columns(6)
                            for col_idx in range(6):
                                idx = start_idx + row + col_idx
                                if idx >= end_idx:
                                    break
                                p = filtered_pieces[idx]
                                with cols[col_idx]:
                                    st.image(p['thumbnail'])
                                    st.caption(f"🏷️ {p.get('ai_name', 'N/A')}")
                                    st.caption(f"🔢 ID: {p.get('original_id', '?')}")
                                    if 'batch' in p:
                                        st.caption(f"📦 {p['batch']}")
                                    if 'cluster_key' in p:
                                        cluster_info = db['clusters'].get(p['cluster_key'], {})
                                        st.caption(f"📊 {cluster_info.get('name', 'N/A')}")
            
            with db_tab2:
                if not db.get('clusters'):
                    st.info("Keine Cluster")
                else:
                    for cluster_key, cluster_data in db['clusters'].items():
                        with st.expander(f"📦 {cluster_data['name']} ({cluster_data['piece_count']} Teile)"):
                            st.write(f"**Datum:** {cluster_data['created'][:10]}")
                            preview_cols = st.columns(6)
                            for i, p in enumerate(cluster_data['pieces'][:6]):
                                with preview_cols[i]:
                                    st.image(p['thumbnail'])
                                    st.caption(p.get('ai_name', 'N/A'))
            
            with db_tab3:
                st.subheader("🗑️ Editor")
                st.warning("⚠️ Permanent!")
                
                # Teile zur DB hinzufügen
                with st.expander("➕ Teile zur DB hinzufügen", expanded=False):
                    if not st.session_state.pieces:
                        st.info("Erst Bilder analysieren!")
                    else:
                        st.markdown("**Aktuelle Session-Teile zur Datenbank hinzufügen:**")
                        
                        active_current = [p for p in st.session_state.pieces if not p['deleted']]
                        
                        if 'add_to_db_indices' not in st.session_state:
                            st.session_state.add_to_db_indices = set()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Alle auswählen", use_container_width=True, key="select_all_btn"):
                                st.session_state.add_to_db_indices = set(p['id'] for p in active_current)
                                st.session_state.force_checkbox_update = not st.session_state.get('force_checkbox_update', False)
                        with col2:
                            if st.button("❌ Alle abwählen", use_container_width=True, key="deselect_all_btn"):
                                st.session_state.add_to_db_indices = set()
                                st.session_state.force_checkbox_update = not st.session_state.get('force_checkbox_update', False)
                        
                        st.markdown("---")
                        
                        force_key = st.session_state.get('force_checkbox_update', False)
                        st.markdown(f"**Teile auswählen ({len(st.session_state.add_to_db_indices)}/{len(active_current)} ausgewählt):**")
                        
                        for row_start in range(0, len(active_current), 6):
                            cols = st.columns(6)
                            for col_idx in range(6):
                                idx = row_start + col_idx
                                if idx >= len(active_current):
                                    break
                                
                                p = active_current[idx]
                                with cols[col_idx]:
                                    st.image(p['thumbnail'])
                                    st.caption(f"{p.get('ai_name', 'N/A')}")
                                    
                                    checkbox_value = p['id'] in st.session_state.add_to_db_indices
                                    checkbox_key = f"dbadd_{p['id']}_{force_key}"
                                    
                                    checked = st.checkbox(
                                        f"ID {p['id']}", 
                                        value=checkbox_value,
                                        key=checkbox_key
                                    )
                                    
                                    if checked and p['id'] not in st.session_state.add_to_db_indices:
                                        st.session_state.add_to_db_indices.add(p['id'])
                                    elif not checked and p['id'] in st.session_state.add_to_db_indices:
                                        st.session_state.add_to_db_indices.discard(p['id'])
                        
                        st.markdown("---")
                        
                        # Cluster zuweisen
                        st.markdown("**Optional: Zu Cluster zuweisen**")
                        assign_to_cluster = st.checkbox("Zu bestehendem Cluster zuweisen?")
                        
                        target_cluster = None
                        if assign_to_cluster and db.get('clusters'):
                            cluster_options = {
                                f"{v['name']} ({v['piece_count']} Teile)": k 
                                for k, v in db['clusters'].items()
                            }
                            selected = st.selectbox("Cluster wählen:", list(cluster_options.keys()))
                            target_cluster = cluster_options[selected]
                        
                        if st.session_state.add_to_db_indices:
                            st.success(f"📌 {len(st.session_state.add_to_db_indices)} Teile markiert")
                            
                            if st.button("➕ Zur DB hinzufügen", type="primary", use_container_width=True):
                                added_count = 0
                                for p_id in st.session_state.add_to_db_indices:
                                    piece = next((p for p in active_current if p['id'] == p_id), None)
                                    if piece:
                                        piece_data = {
                                            'features': piece['features'],
                                            'thumbnail': piece['thumbnail'],
                                            'original_id': piece['id'],
                                            'area': piece['area'],
                                            'ai_name': piece.get('ai_name', 'Unbenannt'),
                                            'batch': piece.get('batch', '')
                                        }
                                        
                                        if target_cluster and target_cluster in db['clusters']:
                                            piece_data['cluster_key'] = target_cluster
                                            db['clusters'][target_cluster]['pieces'].append(piece_data)
                                            db['clusters'][target_cluster]['piece_count'] += 1
                                        
                                        db['pieces'].append(piece_data)
                                        added_count += 1
                                
                                save_database(db)
                                st.session_state.add_to_db_indices = set()
                                st.success(f"✓ {added_count} Teile zur DB hinzugefügt!")
                                st.rerun()
                        else:
                            st.info("Keine Teile ausgewählt")
                
                st.markdown("---")
                
                if not db['pieces'] and not db.get('clusters'):
                    st.info("DB leer")
                else:
                    with st.expander("📋 Einzelne Teile löschen", expanded=False):
                        if not db['pieces']:
                            st.info("Keine Teile")
                        else:
                            if 'db_delete_indices' not in st.session_state:
                                st.session_state.db_delete_indices = set()
                            
                            for row_start in range(0, min(24, len(db['pieces'])), 6):
                                cols = st.columns(6)
                                for col_idx in range(6):
                                    idx = row_start + col_idx
                                    if idx >= len(db['pieces']):
                                        break
                                    
                                    with cols[col_idx]:
                                        p = db['pieces'][idx]
                                        st.image(p['thumbnail'])
                                        is_checked = st.checkbox(
                                            f"#{idx}", 
                                            value=idx in st.session_state.db_delete_indices, 
                                            key=f"dbdel_{idx}"
                                        )
                                        
                                        if is_checked:
                                            st.session_state.db_delete_indices.add(idx)
                                        else:
                                            st.session_state.db_delete_indices.discard(idx)
                            
                            if st.session_state.db_delete_indices:
                                if st.button("🗑️ Markierte löschen", type="primary"):
                                    for idx in sorted(st.session_state.db_delete_indices, reverse=True):
                                        if idx < len(db['pieces']):
                                            del db['pieces'][idx]
                                    save_database(db)
                                    st.session_state.db_delete_indices = set()
                                    st.rerun()
                    
                    with st.expander("📦 Cluster löschen", expanded=False):
                        if not db.get('clusters'):
                            st.info("Keine Cluster")
                        else:
                            for cluster_key in list(db['clusters'].keys()):
                                cluster_data = db['clusters'][cluster_key]
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"📦 **{cluster_data['name']}** ({cluster_data['piece_count']})")
                                with col2:
                                    if st.button("🗑️", key=f"delc_{cluster_key}"):
                                        del db['clusters'][cluster_key]
                                        save_database(db)
                                        st.rerun()
                    
                    with st.expander("🔥 Alles löschen", expanded=False):
                        confirm = st.text_input("'LÖSCHEN' eingeben:", key="confirm_all")
                        if st.button("🔥 ALLES LÖSCHEN"):
                            if confirm == "LÖSCHEN":
                                save_database({'pieces': [], 'clusters': {}, 'version': FEATURE_VERSION})
                                st.rerun()

        # TAB 5: QR-SCANNER
        with tab5:
            st.header("📷 QR-Code Scanner")
            st.markdown("Scanne QR-Codes von Labels um Teile in der Datenbank zu finden")
            
            qr_input = st.text_input(
                "QR-Code eingeben oder scannen:",
                placeholder="shardmind://piece/5 oder einfach die ID",
                key="qr_scanner_input"
            )
            
            if qr_input:
                piece_id = None
                batch = None
                
                if "shardmind://piece/" in qr_input:
                    try:
                        parts = qr_input.replace("shardmind://piece/", "").split("?")
                        piece_id = int(parts[0])
                        if len(parts) > 1 and "batch=" in parts[1]:
                            batch = parts[1].split("batch=")[1]
                    except:
                        st.error("Ungültiger QR-Code")
                else:
                    try:
                        piece_id = int(qr_input)
                    except:
                        st.error("Bitte ID oder QR-Code eingeben")
                
                if piece_id is not None:
                    session_piece = next(
                        (p for p in st.session_state.pieces if p['id'] == piece_id and not p['deleted']), 
                        None
                    )
                    db_pieces = [p for p in db['pieces'] if p.get('original_id') == piece_id]
                    
                    if session_piece or db_pieces:
                        st.success(f"✓ Teil gefunden: ID {piece_id}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if session_piece:
                                st.markdown("### 📌 In aktueller Session:")
                                st.image(session_piece['thumbnail'], width=300)
                                st.write(f"**Name:** {session_piece.get('ai_name', 'N/A')}")
                                st.write(f"**ID:** {session_piece['id']}")
                                if 'batch' in session_piece:
                                    st.write(f"**Batch:** {session_piece['batch']}")
                                if 'cluster' in session_piece and session_piece['cluster'] >= 0:
                                    cluster_name = st.session_state.cluster_names.get(
                                        session_piece['cluster'], 
                                        f"Cluster_{session_piece['cluster']}"
                                    )
                                    st.write(f"**Gruppe:** {cluster_name}")
                                
                                if st.button("🔍 Matches zeigen", key="qr_show_matches"):
                                    st.session_state.selected_id = session_piece['id']
                                    st.rerun()
                        
                        with col2:
                            if db_pieces:
                                st.markdown(f"### 💾 In Datenbank ({len(db_pieces)}x):")
                                for i, db_piece in enumerate(db_pieces[:3]):
                                    st.image(db_piece['thumbnail'], width=150)
                                    st.caption(f"🏷️ {db_piece.get('ai_name', 'N/A')}")
                                    if 'batch' in db_piece:
                                        st.caption(f"📦 {db_piece['batch']}")
                                    if 'cluster_key' in db_piece:
                                        cluster_info = db['clusters'].get(db_piece['cluster_key'], {})
                                        st.caption(f"📊 {cluster_info.get('name', 'N/A')}")
                                    st.markdown("---")
                    else:
                        st.warning(f"❌ Teil mit ID {piece_id} nicht gefunden")

        # TAB 6: VERWALTEN
        with tab6:
            st.header("Verwalten")
            m_cols = st.columns(6)
            for i, p in enumerate(st.session_state.pieces):
                with m_cols[i % 6]:
                    if p['deleted']:
                        st.markdown(
                            f'<div style="opacity: 0.3; border: 2px solid red; padding: 3px;">'
                            f'<img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;">'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        if st.button(f"↩️ #{p['id']}", key=f"rev_{p['id']}", use_container_width=True):
                            p['deleted'] = False
                            st.rerun()
                    else:
                        st.image(p['thumbnail'])
                        st.caption(f"ID: {p['id']}")
                        st.caption(f"🤖 {p.get('ai_name', 'N/A')}")
                        if st.button(f"❌", key=f"del_{p['id']}", use_container_width=True):
                            p['deleted'] = True
                            st.rerun()
    else:
        st.info("👈 Bilder hochladen")


if __name__ == "__main__":
    main()
