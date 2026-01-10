import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import io
from PIL import Image
import base64
import pickle
from pathlib import Path
from datetime import datetime

# ============================================================================
# KONFIGURATION & DB
# ============================================================================
DB_PATH = Path("puzzle_database_v3.pkl")
FEATURE_VERSION = 3


def load_database():
    if DB_PATH.exists():
        try:
            with open(DB_PATH, 'rb') as f:
                db = pickle.load(f)
                if db.get('version') != FEATURE_VERSION:
                    return {
                        'pieces': [], 
                        'clusters': {},
                        'version': FEATURE_VERSION
                    }
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


# ============================================================================
# SEGMENTIERUNG
# ============================================================================
def is_valid_puzzle_piece(contour, roi, mask_roi, image_shape):
    """Validiert ob echtes Puzzleteil"""
    area = cv2.contourArea(contour)
    if area < 200:
        return False
    
    h, w = image_shape[:2]
    if area > (h * w * 0.85):
        return False
    
    x, y, wb, hb = cv2.boundingRect(contour)
    aspect = wb / (hb + 1e-6)
    if aspect < 0.08 or aspect > 12:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (area + 1e-6)
    if compactness > 120:
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
    solidity = area / (hull_area + 1e-6)
    if solidity < 0.5:
        return False
    
    return True


def segment_pieces_robust(image, min_area=100, start_id=0):
    """
    Robuste Segmentierung mit Start-ID für Multi-Bild-Support
    """
    h, w = image.shape[:2]
    
    pad = 30
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    blurred = cv2.GaussianBlur(padded, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    thresh1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh = cv2.bitwise_or(thresh1, thresh2)
    thresh = cv2.bitwise_or(thresh, thresh3)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=5)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces = []
    piece_counter = start_id  # Starte mit übergebener ID
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        if area < min_area:
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

        pieces.append({
            'id': piece_counter,  # Eindeutige ID
            'contour': c,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': area,
            'deleted': False
        })
        piece_counter += 1
    
    return pieces


# ============================================================================
# FEATURES
# ============================================================================
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
    
    sig = np.interp(
        np.linspace(-np.pi, np.pi, 180), 
        angles[s_idx], 
        dists[s_idx], 
        period=2 * np.pi
    )
    sig = sig / (sig.max() + 1e-6)

    lab = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2HSV)
    
    lab_mean, lab_std = cv2.meanStdDev(lab, mask=p['mask'])
    hsv_mean, hsv_std = cv2.meanStdDev(hsv, mask=p['mask'])
    
    color_features = np.concatenate([
        lab_mean.flatten(), 
        lab_std.flatten(),
        hsv_mean.flatten(),
        hsv_std.flatten()
    ])
    
    return {
        'shape': sig, 
        'color': color_features
    }


def calculate_score(f1, f2):
    dist_c = np.linalg.norm(f1['color'] - f2['color'])
    score_c = max(0, 100 - (dist_c / 3.0))
    
    s1, s2 = f1['shape'], f2['shape']
    best_diff = min([
        np.mean((s1 - np.roll(s2, r)) ** 2) 
        for r in range(0, 180, 3)
    ])
    
    if best_diff < 0.02:
        score_s = 95 + (0.02 - best_diff) * 250
    else:
        score_s = max(0, 95 * (1 - best_diff * 6))
    
    final_score = (0.4 * score_c) + (0.6 * score_s)
    
    return min(100, max(0, final_score))


# ============================================================================
# VERBESSERTES CLUSTERING - KOMBINATION AUS FORM + FARBE
# ============================================================================
def cluster_pieces_smart(active_pieces, distance_threshold=1.5):
    """
    Hierarchisches Clustering mit optimierten Parametern
    """
    if len(active_pieces) < 2:
        return [-1] * len(active_pieces)
    
    features = []
    for p in active_pieces:
        # WICHTIG: Nur Farbe für Clustering (Form zu variabel)
        # Aber mit mehr Gewicht auf LAB
        combined = p['features']['color']
        features.append(combined)
    
    X = np.array(features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hierarchisches Clustering mit Average Linkage (robuster)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',  # Average statt Ward für variable Größen
        metric='euclidean'
    )
    
    labels = clustering.fit_predict(X_scaled)
    
    return labels


def get_cluster_color(cluster_id):
    if cluster_id == -1:
        return "rgb(180, 180, 180)"
    
    hue = int((cluster_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


# ============================================================================
# CLUSTER-MANAGEMENT
# ============================================================================
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
            'area': p['area']
        }
        db['clusters'][cluster_key]['pieces'].append(piece_data)
        db['pieces'].append(piece_data)
    
    save_database(db)
    return len(cluster_pieces)


# ============================================================================
# MAIN UI
# ============================================================================
def main():
    st.set_page_config(page_title="Puzzle Master Pro", layout="wide")

    if 'pieces' not in st.session_state: 
        st.session_state.pieces = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}
    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True

    db = load_database()

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("⚙️ Steuerung")
        
        files = st.file_uploader(
            "📤 Bilder hochladen", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        st.markdown("---")
        st.subheader("🔧 Parameter")
        
        min_area = st.slider(
            "Min. Teilgröße", 
            50, 1000, 200, 10,
            help="px² - Höher = weniger Artefakte"
        )
        
        cluster_dist = st.slider(
            "Cluster-Distanz", 
            0.5, 4.0, 1.5, 0.1,
            help="Höher = lockere Cluster (1.0-2.0 empfohlen)"
        )

        st.markdown("---")
        
        if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
            if not files:
                st.warning("⚠️ Bitte Bilder hochladen!")
            else:
                with st.spinner("Verarbeite Bilder..."):
                    all_found = []
                    progress = st.progress(0)
                    
                    current_id = 0  # ID-Counter für alle Bilder
                    
                    for i, f in enumerate(files):
                        img = cv2.imdecode(
                            np.asarray(bytearray(f.read()), dtype=np.uint8), 
                            cv2.IMREAD_COLOR
                        )
                        # Übergebe aktuelle ID und erhalte Teile
                        pieces = segment_pieces_robust(img, min_area, start_id=current_id)
                        all_found.extend(pieces)
                        
                        # Update ID für nächstes Bild
                        if pieces:
                            current_id = max(p['id'] for p in pieces) + 1
                        
                        progress.progress((i + 1) / len(files))

                    # Feature-Extraktion
                    valid = []
                    for p in all_found:
                        feat = get_features(p)
                        if feat:
                            p['features'] = feat
                            valid.append(p)

                    st.session_state.pieces = valid
                    st.session_state.cluster_names = {}
                    st.session_state.show_tutorial = False
                    st.success(f"✓ {len(valid)} Teile aus {len(files)} Bildern erkannt!")
                    st.rerun()

        if st.button("🗑️ Alles löschen", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("💾 Datenbank")
        st.metric("Teile", len(db['pieces']))
        st.metric("Cluster", len(db.get('clusters', {})))
        
        if st.button("❓ Tutorial anzeigen", use_container_width=True):
            st.session_state.show_tutorial = True
            st.rerun()

    # ========== TUTORIAL / STARTSEITE ==========
    if st.session_state.show_tutorial and not st.session_state.pieces:
        st.title("🧩 Puzzle Master Pro - Tutorial")
        
        st.markdown("### 🎯 Willkommen!")
        st.info("Diese App erkennt Puzzle-Teile, gruppiert sie automatisch und hilft dir, passende Teile zu finden.")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Schritt 1: Bilder vorbereiten")
            st.markdown("""
            **Wichtig für beste Ergebnisse:**
            - ✅ Einfarbiger Hintergrund (weiß, schwarz oder grau)
            - ✅ Gleichmäßige Beleuchtung
            - ✅ Teile berühren sich nicht
            - ✅ 4-6 Bilder mit je 10-20 Teilen
            - ❌ Keine Schatten oder Reflexionen
            """)
            
            st.markdown("### 📤 Schritt 2: Hochladen & Analysieren")
            st.markdown("""
            1. **Sidebar** → "Bilder hochladen" klicken
            2. **Mehrere Bilder** auswählen (Strg/Cmd + Klick)
            3. **Parameter einstellen:**
               - Min. Teilgröße: 200-400 px²
               - Cluster-Distanz: 1.0-2.0
            4. **"Analyse starten"** klicken
            """)
        
        with col2:
            st.markdown("### 🎯 Schritt 3: Cluster verwalten")
            st.markdown("""
            Nach der Analyse:
            1. **Tab "Galerie"**: Alle erkannten Teile sehen
            2. **Tab "Cluster-Manager"**: 
               - Cluster werden nach Farbe/Form gruppiert
               - Jeden Cluster benennen (z.B. "Himmel", "Rand")
               - Mit 💾 in Datenbank speichern
            3. **"Clustering neu berechnen"** falls nötig
            """)
            
            st.markdown("### 🔍 Schritt 4: Matches finden")
            st.markdown("""
            1. **Tab "Galerie"**: Teil auswählen (🔍 klicken)
            2. **Tab "Matching"**: Top 10 passende Teile sehen
            3. **Tab "Datenbank"**: 
               - Mit gespeicherten Teilen vergleichen
               - "Vergleichs-Modus" aktivieren
            """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Tipps & Tricks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎨 Clustering**")
            st.markdown("""
            - **Alle einzeln?** → Distanz erhöhen (2.0-3.0)
            - **Zu viele Cluster?** → Distanz senken (0.8-1.2)
            - **Falsche Gruppen?** → "Neu berechnen"
            """)
        
        with col2:
            st.markdown("**🗑️ Fehler beheben**")
            st.markdown("""
            - **Weiße Flächen erkannt?** → Im Tab "Verwalten" löschen
            - **Nach Löschen neu clustern!**
            - **Min. Teilgröße erhöhen** → weniger Artefakte
            """)
        
        with col3:
            st.markdown("**💾 Datenbank**")
            st.markdown("""
            - Cluster einzeln speichern
            - Später mit neuen Teilen vergleichen
            - Tab "Datenbank" → Einzelne Teile löschen
            """)
        
        st.markdown("---")
        
        st.success("🚀 Bereit? Lade Bilder in der Sidebar hoch und klicke auf 'Analyse starten'!")
        
        st.markdown("---")
        st.caption("💡 Tipp: Dieses Tutorial kann jederzeit über den Button in der Sidebar wieder angezeigt werden.")

    # ========== HAUPTFENSTER MIT TEILEN ==========
    elif st.session_state.pieces:
        # Clustering
        active_pieces = [p for p in st.session_state.pieces if not p['deleted']]
        
        if len(active_pieces) > 1:
            labels = cluster_pieces_smart(active_pieces, cluster_dist)
            for i, p in enumerate(active_pieces): 
                p['cluster'] = labels[i]
        else:
            for p in active_pieces:
                p['cluster'] = -1
        
        # Statistiken
        cluster_ids = set([p.get('cluster', -1) for p in active_pieces])
        n_clusters = len([c for c in cluster_ids if c >= 0])
        n_noise = sum(1 for p in active_pieces if p.get('cluster', -1) == -1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Teile", len(active_pieces))
        col2.metric("🎯 Cluster", n_clusters)
        col3.metric("⚪ Einzeln", n_noise)
        
        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Galerie", 
            "📦 Cluster-Manager",
            "🔍 Matching", 
            "💾 Datenbank",
            "❌ Verwalten"
        ])

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
                    
                    if st.button(f"🔍", key=f"sel_{p['id']}", use_container_width=True):
                        st.session_state.selected_id = p['id']
                        st.rerun()

        with tab2:
            st.header("📦 Cluster verwalten")
            
            cluster_ids = sorted([c for c in cluster_ids if c >= 0])
            
            if not cluster_ids:
                st.warning("⚠️ Keine Cluster gefunden!")
                st.info("💡 Erhöhe die **Cluster-Distanz** in der Sidebar (z.B. auf 2.0-3.0)")
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
                        
                        st.markdown("**Teile:**")
                        preview_cols = st.columns(min(6, len(cluster_pieces)))
                        for i, p in enumerate(cluster_pieces[:6]):
                            with preview_cols[i]:
                                st.image(p['thumbnail'], caption=f"ID: {p['id']}")
                        
                        if len(cluster_pieces) > 6:
                            st.caption(f"... und {len(cluster_pieces) - 6} weitere")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button(
                                f"💾 '{cluster_name}' speichern", 
                                key=f"save_{cluster_id}",
                                use_container_width=True
                            ):
                                count = save_cluster_to_db(active_pieces, cluster_id, cluster_name, db)
                                st.success(f"✓ {count} Teile gespeichert!")
                                st.rerun()
                        
                        with col2:
                            st.metric("Teile", len(cluster_pieces))

        with tab3:
            if 'selected_id' in st.session_state:
                target = next((p for p in active_pieces if p['id'] == st.session_state.selected_id), None)
                
                if target:
                    st.header(f"Matches für Teil #{target['id']}")
                    
                    col_l, col_r = st.columns([1, 4])
                    
                    with col_l:
                        st.markdown("**Ausgewählt:**")
                        st.image(target['thumbnail'], width=200)
                        cluster_id = target.get('cluster', -1)
                        if cluster_id >= 0:
                            cluster_name = st.session_state.cluster_names.get(cluster_id, f"Cluster_{cluster_id}")
                            st.info(f"Cluster: {cluster_name}")
                    
                    with col_r:
                        st.markdown("**Top 10 Matches:**")
                        
                        matches = sorted([
                            (calculate_score(target['features'], p['features']), p)
                            for p in active_pieces if p['id'] != target['id']
                        ], key=lambda x: x[0], reverse=True)
                        
                        for row in range(2):
                            m_cols = st.columns(5)
                            for col in range(5):
                                idx = row * 5 + col
                                if idx >= len(matches):
                                    break
                                
                                score, p = matches[idx]
                                with m_cols[col]:
                                    st.image(p['thumbnail'], caption=f"ID {p['id']}")
                                    st.progress(score / 100)
                                    st.markdown(f"**{score:.1f}%**")
                else:
                    st.warning("Teil gelöscht")
            else:
                st.info("👈 Wähle ein Teil in der Galerie")

        with tab4:
            st.header("💾 Datenbank verwalten")
            
            db_tab1, db_tab2, db_tab3 = st.tabs(["📋 Alle Teile", "📦 Cluster", "🗑️ DB-Editor"])
            
            with db_tab1:
                st.subheader(f"Alle Teile ({len(db['pieces'])})")
                
                if not db['pieces']:
                    st.info("Datenbank leer")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_per_page = st.selectbox("Pro Seite", [12, 24, 48], index=1)
                    with col2:
                        if 'selected_id' in st.session_state:
                            compare_mode = st.checkbox("Vergleichs-Modus")
                        else:
                            compare_mode = False
                    
                    total_pages = (len(db['pieces']) - 1) // show_per_page + 1
                    page = st.slider("Seite", 1, total_pages, 1)
                    
                    start_idx = (page - 1) * show_per_page
                    end_idx = min(start_idx + show_per_page, len(db['pieces']))
                    
                    if compare_mode:
                        target = next((p for p in active_pieces if p['id'] == st.session_state.selected_id), None)
                        if target:
                            db_with_scores = []
                            for db_piece in db['pieces'][start_idx:end_idx]:
                                score = calculate_score(target['features'], db_piece['features'])
                                db_with_scores.append((score, db_piece))
                            
                            db_with_scores.sort(key=lambda x: x[0], reverse=True)
                            
                            for row in range(0, len(db_with_scores), 6):
                                cols = st.columns(6)
                                for col_idx in range(6):
                                    idx = row + col_idx
                                    if idx >= len(db_with_scores):
                                        break
                                    
                                    score, p = db_with_scores[idx]
                                    with cols[col_idx]:
                                        st.image(p['thumbnail'])
                                        st.progress(score / 100)
                                        st.markdown(f"**{score:.1f}%**")
                    else:
                        for row in range(0, end_idx - start_idx, 6):
                            cols = st.columns(6)
                            for col_idx in range(6):
                                idx = start_idx + row + col_idx
                                if idx >= end_idx:
                                    break
                                
                                p = db['pieces'][idx]
                                with cols[col_idx]:
                                    st.image(p['thumbnail'])
                                    st.caption(f"#{idx}")
            
            with db_tab2:
                st.subheader(f"Cluster ({len(db.get('clusters', {}))})")
                
                if not db.get('clusters'):
                    st.info("Keine Cluster gespeichert")
                else:
                    for cluster_key, cluster_data in db['clusters'].items():
                        with st.expander(
                            f
