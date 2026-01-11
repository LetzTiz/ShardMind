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

DB_PATH = Path("puzzle_database_v3.pkl")
FEATURE_VERSION = 3

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

def image_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def generate_ai_name(piece):
    """Generiert KI-basierten Namen basierend auf Farbe und Form"""
    # Farb-Analyse
    lab = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(piece['thumbnail'], cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=piece['mask'])
    
    # Farbname bestimmen
    h, s, v = mean_hsv[:3]
    if s < 50:
        color = "Grau" if v < 150 else "Weiß"
    elif h < 15 or h > 165:
        color = "Rot"
    elif h < 35:
        color = "Orange"
    elif h < 75:
        color = "Gelb"
    elif h < 95:
        color = "Grün"
    elif h < 130:
        color = "Cyan"
    else:
        color = "Blau"
    
    # Form-Analyse
    area = piece['area']
    if area < 500:
        size = "Klein"
    elif area < 2000:
        size = "Mittel"
    else:
        size = "Groß"
    
    # Kompaktheit
    perimeter = cv2.arcLength(piece['contour'], True)
    compactness = (perimeter ** 2) / (area + 1e-6)
    
    if compactness < 15:
        shape = "Rund"
    elif compactness < 25:
        shape = "Kompakt"
    else:
        shape = "Komplex"
    
    return f"{size}_{color}_{shape}"

def generate_qr_code(data):
    """Generiert QR-Code als PIL Image"""
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

def create_label_pdf(pieces, filename="puzzle_labels.pdf"):
    """Erstellt PDF mit QR-Codes und Labels"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Layout: 3 Spalten, 7 Zeilen pro Seite
    cols, rows = 3, 7
    cell_width = width / cols
    cell_height = height / rows
    
    x_offset, y_offset = 20, 20
    qr_size = 80
    
    for idx, piece in enumerate(pieces):
        if piece.get('deleted', False):
            continue
        
        # Position berechnen
        col = idx % cols
        row = (idx // cols) % rows
        
        if idx > 0 and idx % (cols * rows) == 0:
            c.showPage()  # Neue Seite
        
        x = col * cell_width + x_offset
        y = height - (row + 1) * cell_height + y_offset
        
        # QR-Code generieren
        qr_data = f"ID:{piece['id']}|Name:{piece.get('ai_name', 'Unbenannt')}"
        qr_img = generate_qr_code(qr_data)
        
        # QR-Code in PDF einfügen
        qr_buffer = io.BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)
        
        c.drawImage(ImageReader(qr_buffer), x, y, width=qr_size, height=qr_size)
        
        # Text
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + qr_size + 10, y + qr_size - 15, f"ID: {piece['id']}")
        c.setFont("Helvetica", 9)
        c.drawString(x + qr_size + 10, y + qr_size - 30, piece.get('ai_name', 'Unbenannt'))
        
        if 'cluster' in piece and piece['cluster'] >= 0:
            c.drawString(x + qr_size + 10, y + qr_size - 45, f"Cluster: {piece['cluster']}")
    
    c.save()
    buffer.seek(0)
    return buffer

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

def segment_pieces_robust(image, min_area=100, start_id=0):
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

        piece = {
            'id': start_id + len(pieces),
            'contour': c,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': cv2.contourArea(c),
            'deleted': False
        }
        piece['ai_name'] = generate_ai_name(piece)
        pieces.append(piece)
    
    return pieces

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

def main():
    st.set_page_config(
        page_title="ShardMind - KI Scherben-Analyse", 
        page_icon="🧠",
        layout="wide"
    )

    if 'pieces' not in st.session_state:
        st.session_state.pieces = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}
    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True

    db = load_database()

    with st.sidebar:
        st.title("🧠 ShardMind")
        st.caption("KI-gestützte Scherben-Analyse")
        
        files = st.file_uploader("📤 Bilder hochladen", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("🔧 Parameter")
        min_area = st.slider("Min. Teilgröße", 50, 1000, 200, 10)
        cluster_dist = st.slider("Cluster-Distanz", 0.1, 10.0, 1.5, 0.1)

        st.markdown("---")
        
        if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
            if not files:
                st.warning("⚠️ Bitte Bilder hochladen!")
            else:
                with st.spinner("Verarbeite Bilder..."):
                    all_found = []
                    progress = st.progress(0)
                    
                    # Wichtig: ID-Counter muss über ALLE Dateien durchlaufen
                    current_id = 0
                    
                    for i, f in enumerate(files):
                        try:
                            # Bild laden
                            img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                            
                            # Segmentiere mit aktuellem ID-Counter
                            pieces = segment_pieces_robust(img, min_area, start_id=current_id)
                            
                            # Erweitere Liste
                            all_found.extend(pieces)
                            
                            # Update ID-Counter für nächstes Bild
                            if pieces:
                                current_id = max(p['id'] for p in pieces) + 1
                            
                            progress.progress((i + 1) / len(files))
                        
                        except Exception as e:
                            st.error(f"Fehler bei Bild {i+1}: {e}")
                            continue

                    # Feature-Extraktion für alle gesammelten Teile
                    valid = []
                    for p in all_found:
                        feat = get_features(p)
                        if feat:
                            p['features'] = feat
                            valid.append(p)

                    st.session_state.pieces = valid
                    st.session_state.cluster_names = {}
                    st.session_state.show_tutorial = False
                    
                    # Erfolgs-Meldung mit Details
                    st.success(f"✓ {len(valid)} Teile aus {len(files)} Bildern erkannt! (KI-Namen generiert)")
                    
                    # Debug-Info
                    if len(valid) > 0:
                        st.info(f"IDs: {valid[0]['id']} bis {valid[-1]['id']}")
                    
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
        
        # PDF Download - IMMER sichtbar
        st.markdown("---")
        st.subheader("🖨️ Drucken")
        
        if st.session_state.pieces:
            active_for_print = [p for p in st.session_state.pieces if not p['deleted']]
            if active_for_print:
                st.caption(f"{len(active_for_print)} Teile bereit")
                if st.button("📄 PDF erstellen", use_container_width=True):
                    with st.spinner("Erstelle PDF..."):
                        pdf_buffer = create_label_pdf(active_for_print)
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

    if st.session_state.show_tutorial and not st.session_state.pieces:
        st.title("🧠 ShardMind - Tutorial")
        st.markdown("### KI-gestützte Scherben-Analyse")
        
        st.info("""
        **ShardMind** nutzt Computer Vision & KI zur Analyse von Puzzle-Teilen und archäologischen Scherben:
        - 🤖 **KI-Benennung**: Automatische Beschreibung (Farbe, Größe, Form)
        - 🎨 **Auto-Clustering**: Gruppiert ähnliche Teile
        - 🔍 **Smart Matching**: Findet passende Teile (bis 100% Genauigkeit)
        - 💾 **Datenbank**: Speichert & verwaltet Teile mit Clustern
        - 🖨️ **QR-Code-PDF**: Drucke Labels für physische Objekte
        """)
        
        st.markdown("---")
        
        tabs = st.tabs(["📸 Vorbereitung", "🎯 Analyse", "💾 Datenbank", "🖨️ Export"])
        
        with tabs[0]:
            st.markdown("### 📸 Schritt 1: Bilder vorbereiten")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Anforderungen:**")
                st.markdown("""
                - Einfarbiger Hintergrund (weiß/schwarz/grau)
                - Gleichmäßige Beleuchtung
                - Teile berühren sich nicht
                - 4-6 Bilder, je 10-20 Teile
                - Min. 1920x1080 Auflösung
                """)
            with col2:
                st.markdown("**🚀 Upload & Start:**")
                st.markdown("""
                1. Sidebar → "📤 Bilder hochladen"
                2. Mehrere auswählen (Strg/Cmd + Klick)
                3. Parameter: Min. 200, Cluster 1.0-2.0
                4. "🚀 Analyse starten"
                """)
        
        with tabs[1]:
            st.markdown("### 🎯 Schritt 2: Teile analysieren")
            st.markdown("""
            **Tab 'Galerie':**
            - Farbige Rahmen = Cluster-Zugehörigkeit
            - ID + KI-Name unter jedem Teil
            - 🔍 Klick = Auswahl für Matching
            
            **Tab 'Cluster-Manager':**
            1. Cluster benennen (z.B. "Himmel", "Rand")
            2. Vorschau der Teile
            3. 💾 Speichern in Datenbank
            
            **Tab 'Verwalten':**
            - ❌ Fehlerhafte Teile löschen
            - ↩️ Wiederherstellen
            - Dann: "Neu berechnen"!
            """)
        
        with tabs[2]:
            st.markdown("### 💾 Schritt 3: Datenbank nutzen")
            st.markdown("""
            **Teile zur DB hinzufügen:**
            - Tab "Datenbank" → "🗑️ Editor"
            - Expander "➕ Teile hinzufügen"
            - Teile aus aktueller Session auswählen
            - Cluster zuweisen (optional)
            - "Zur DB hinzufügen" klicken
            
            **Alle Teile durchsuchen:**
            - Zeigt ID, KI-Name, Cluster
            - Pagination (12/24/48 pro Seite)
            
            **Cluster verwalten:**
            - Alle gespeicherten Cluster
            - Vorschau der Teile
            - Info: Name, Datum, Anzahl
            """)
        
        with tabs[3]:
            st.markdown("### 🖨️ Schritt 4: QR-Codes drucken")
            st.markdown("""
            **PDF erstellen:**
            1. Sidebar → "🖨️ Drucken"
            2. "📄 PDF erstellen"
            3. "⬇️ PDF herunterladen"
            4. Drucken & ausschneiden
            
            **PDF enthält:**
            - QR-Code (scannbar mit Handy)
            - ID + KI-Name + Cluster
            - Layout: 3×7 pro A4-Seite
            
            **Verwendung:**
            - Labels auf physische Teile kleben
            - QR-Code scannen → ID sofort erkannt
            """)
        
        st.markdown("---")
        st.success("🚀 Bereit? Lade Bilder hoch!")

    elif st.session_state.pieces:
        active_pieces = [p for p in st.session_state.pieces if not p['deleted']]
        
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
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Teile", len(active_pieces))
        col2.metric("🎯 Cluster", n_clusters)
        col3.metric("⚪ Einzeln", n_noise)
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Galerie", "📦 Cluster-Manager", "🔍 Matching", "💾 Datenbank", "❌ Verwalten"])

        with tab1:
            st.header("Erkannte Teile")
            if st.button("🔄 Clustering neu berechnen"):
                labels = cluster_pieces_smart(active_pieces, cluster_dist)
                for i, p in enumerate(active_pieces):
                    p['cluster'] = labels[i]
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
                        cluster_name = st.text_input("Name:", value=default_name, key=f"name_{cluster_id}")
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
                            if st.button(f"💾 '{cluster_name}' speichern", key=f"save_{cluster_id}", use_container_width=True):
                                count = save_cluster_to_db(active_pieces, cluster_id, cluster_name, db)
                                st.success(f"✓ {count} Teile!")
                                st.rerun()
                        with col2:
                            st.metric("Teile", len(cluster_pieces))

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
                        matches = sorted([(calculate_score(target['features'], p['features']), p) for p in active_pieces if p['id'] != target['id']], key=lambda x: x[0], reverse=True)
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

        with tab4:
            st.header("💾 Datenbank")
            db_tab1, db_tab2, db_tab3 = st.tabs(["📋 Alle Teile", "📦 Cluster", "🗑️ Editor"])
            
            with db_tab1:
                if not db['pieces']:
                    st.info("DB leer")
                else:
                    show_per_page = st.selectbox("Pro Seite", [12, 24, 48], index=1)
                    total_pages = max(1, (len(db['pieces']) - 1) // show_per_page + 1)
                    page = st.slider("Seite", 1, total_pages, 1) if total_pages > 1 else 1
                    start_idx = (page - 1) * show_per_page
                    end_idx = min(start_idx + show_per_page, len(db['pieces']))
                    
                    for row in range(0, end_idx - start_idx, 6):
                        cols = st.columns(6)
                        for col_idx in range(6):
                            idx = start_idx + row + col_idx
                            if idx >= end_idx:
                                break
                            p = db['pieces'][idx]
                            with cols[col_idx]:
                                st.image(p['thumbnail'])
                                st.caption(f"#{idx} | {p.get('ai_name', 'N/A')}")
                                if 'cluster_key' in p:
                                    cluster_info = db['clusters'].get(p['cluster_key'], {})
                                    st.caption(f"📦 {cluster_info.get('name', 'N/A')}")
            
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
                
                # NEUE FUNKTION: Teile zur DB hinzufügen
                with st.expander("➕ Teile zur DB hinzufügen", expanded=False):
                    if not st.session_state.pieces:
                        st.info("Erst Bilder analysieren!")
                    else:
                        st.markdown("**Aktuelle Session-Teile zur Datenbank hinzufügen:**")
                        
                        active_current = [p for p in st.session_state.pieces if not p['deleted']]
                        
                        if 'add_to_db_indices' not in st.session_state:
                            st.session_state.add_to_db_indices = set()
                        
                        # ALLES AUSWÄHLEN / ABWÄHLEN
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Alle auswählen", use_container_width=True, key="select_all_btn"):
                                # Setze alle IDs
                                st.session_state.add_to_db_indices = set(p['id'] for p in active_current)
                                # Force update durch Session State Flag
                                st.session_state.force_checkbox_update = not st.session_state.get('force_checkbox_update', False)
                        with col2:
                            if st.button("❌ Alle abwählen", use_container_width=True, key="deselect_all_btn"):
                                st.session_state.add_to_db_indices = set()
                                st.session_state.force_checkbox_update = not st.session_state.get('force_checkbox_update', False)
                        
                        st.markdown("---")
                        
                        # Zeige Teile mit Checkboxen - verwende eindeutige Keys mit force_update
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
                                    
                                    # Checkbox mit value aus Session State
                                    checkbox_value = p['id'] in st.session_state.add_to_db_indices
                                    
                                    # Einzigartiger Key der sich bei force_update ändert
                                    checkbox_key = f"dbadd_{p['id']}_{force_key}"
                                    
                                    checked = st.checkbox(
                                        f"ID {p['id']}", 
                                        value=checkbox_value,
                                        key=checkbox_key
                                    )
                                    
                                    # Update Session State basierend auf Checkbox
                                    if checked and p['id'] not in st.session_state.add_to_db_indices:
                                        st.session_state.add_to_db_indices.add(p['id'])
                                    elif not checked and p['id'] in st.session_state.add_to_db_indices:
                                        st.session_state.add_to_db_indices.discard(p['id'])
                        
                        if len(active_current) > 18:
                            st.caption(f"💡 Zeige alle {len(active_current)} Teile - scrollen für mehr")
                        
                        st.markdown("---")
                        
                        # Optional: Cluster zuweisen
                        st.markdown("**Optional: Zu Cluster zuweisen**")
                        assign_to_cluster = st.checkbox("Zu bestehendem Cluster zuweisen?")
                        
                        target_cluster = None
                        if assign_to_cluster and db.get('clusters'):
                            cluster_options = {f"{v['name']} ({v['piece_count']} Teile)": k for k, v in db['clusters'].items()}
                            selected = st.selectbox("Cluster wählen:", list(cluster_options.keys()))
                            target_cluster = cluster_options[selected]
                        
                        # Hinzufügen-Button
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
                                            'ai_name': piece.get('ai_name', 'Unbenannt')
                                        }
                                        
                                        # Zu Cluster hinzufügen falls gewählt
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
                    with st.expander("📋 Einzelne Teile", expanded=False):
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
                                        is_checked = st.checkbox(f"#{idx}", value=idx in st.session_state.db_delete_indices, key=f"dbdel_{idx}")
                                        
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
                    
                    with st.expander("📦 Cluster", expanded=False):
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
                        if st.button("🔥 ALLES"):
                            if confirm == "LÖSCHEN":
                                save_database({'pieces': [], 'clusters': {}, 'version': FEATURE_VERSION})
                                st.rerun()

        with tab5:
            st.header("Verwalten")
            m_cols = st.columns(6)
            for i, p in enumerate(st.session_state.pieces):
                with m_cols[i % 6]:
                    if p['deleted']:
                        st.markdown(f'<div style="opacity: 0.3; border: 2px solid red; padding: 3px;"><img src="data:image/png;base64,{image_to_base64(p["thumbnail"])}" style="width:100%;"></div>', unsafe_allow_html=True)
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
