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
# VERBESSERTE SEGMENTIERUNG MIT FILTER
# ============================================================================
def is_valid_puzzle_piece(contour, roi, mask_roi, image_shape):
    """
    Intelligente Validierung ob Kontur ein echtes Puzzleteil ist
    """
    # 1. Größen-Check
    area = cv2.contourArea(contour)
    if area < 200:  # Zu klein
        return False

    h, w = image_shape[:2]
    if area > (h * w * 0.85):  # Zu groß (>85% = Hintergrund)
        return False

    # 2. Aspect Ratio
    x, y, wb, hb = cv2.boundingRect(contour)
    aspect = wb / (hb + 1e-6)
    if aspect < 0.08 or aspect > 12:  # Extreme Formen
        return False

    # 3. Kompaktheit (gegen sehr zerklüftete Artefakte)
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (area + 1e-6)
    if compactness > 120:  # Zu komplex
        return False

    # 4. WICHTIG: Farb-Varianz (gegen weiße/einfarbige Flächen)
    if roi.shape[0] > 5 and roi.shape[1] > 5:
        # Berechne Standardabweichung in LAB
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        _, std = cv2.meanStdDev(lab, mask=mask_roi)

        # Wenn alle Kanäle sehr niedrige Std haben = einfarbig = kein Puzzleteil
        if np.all(std < 5):  # Sehr geringe Variation
            return False

        # Zusätzlich: Prüfe auf Weiß/Grau
        mean_val = cv2.mean(lab, mask=mask_roi)
        # L-Kanal > 240 und niedrige a,b = Weiß
        if mean_val[0] > 240 and abs(mean_val[1] - 128) < 10 and abs(mean_val[2] - 128) < 10:
            return False

    # 5. Solidity (Verhältnis zur konvexen Hülle)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    if solidity < 0.5:  # Zu viele Löcher/Konkavitäten
        return False

    return True


def segment_pieces_robust(image, min_area=100):
    """
    Robuste Multi-Pass Segmentierung mit intelligentem Filtering
    """
    h, w = image.shape[:2]

    # Erweitertes Padding für Randteile
    pad = 30
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Preprocessing
    blurred = cv2.GaussianBlur(padded, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Multi-Threshold Strategie
    # 1. Adaptive Gaussian
    thresh1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # 2. Adaptive Mean
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # 3. Otsu
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kombiniere alle drei
    thresh = cv2.bitwise_or(thresh1, thresh2)
    thresh = cv2.bitwise_or(thresh, thresh3)

    # Morphologische Operationen
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=5)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Konturen finden
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces = []
    for c in cnts:
        area = cv2.contourArea(c)

        # Basis-Größenfilter
        if area < min_area:
            continue

        x, y, wb, hb = cv2.boundingRect(c)

        # ROI extrahieren
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(padded.shape[1], x + wb + margin), min(padded.shape[0], y + hb + margin)

        roi = padded[y1:y2, x1:x2].copy()

        # Maske für ROI
        mask = np.zeros(padded.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask_roi = mask[y1:y2, x1:x2].copy()

        # WICHTIG: Validierung ob echtes Puzzleteil
        if not is_valid_puzzle_piece(c, roi, mask_roi, padded.shape):
            continue

        pieces.append({
            'id': len(pieces),
            'contour': c,
            'thumbnail': roi,
            'mask': mask_roi,
            'area': area,
            'deleted': False
        })

    return pieces


# ============================================================================
# FEATURES
# ============================================================================
def get_features(p):
    """Extrahiert Features mit Fehlerbehandlung"""
    M = cv2.moments(p['contour'])
    if M['m00'] == 0:
        return None

    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    peri = cv2.arcLength(p['contour'], True)

    # Shape Signature
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

    # Farb-Features (LAB + HSV)
    lab = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(p['thumbnail'], cv2.COLOR_BGR2HSV)

    lab_mean, lab_std = cv2.meanStdDev(lab, mask=p['mask'])
    hsv_mean, hsv_std = cv2.meanStdDev(hsv, mask=p['mask'])

    # Kombiniere für bessere Unterscheidung
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
    """
    Verbessertes Matching mit besserer Skalierung
    """
    # Farb-Distanz (12-dimensional)
    dist_c = np.linalg.norm(f1['color'] - f2['color'])
    # Normalisierung anpassen weil mehr Dimensionen
    score_c = max(0, 100 - (dist_c / 3.0))

    # Form-Matching (Rotation-invariant)
    s1, s2 = f1['shape'], f2['shape']
    best_diff = min([
        np.mean((s1 - np.roll(s2, r)) ** 2)
        for r in range(0, 180, 3)  # Alle 3° testen
    ])

    # Nicht-lineare Skalierung für bessere Differenzierung
    if best_diff < 0.02:  # Sehr ähnlich
        score_s = 95 + (0.02 - best_diff) * 250
    else:
        score_s = max(0, 95 * (1 - best_diff * 6))

    # Gewichtung: Form 60%, Farbe 40%
    final_score = (0.4 * score_c) + (0.6 * score_s)

    return min(100, max(0, final_score))


# ============================================================================
# VERBESSERTES CLUSTERING
# ============================================================================
def cluster_pieces_smart(active_pieces, distance_threshold=0.5):
    """
    Hierarchisches Clustering mit kombinierten Features
    Besser als DBSCAN für variable Dichten
    """
    if len(active_pieces) < 2:
        return [-1] * len(active_pieces)

    # Kombiniere Form + Farbe für Clustering
    features = []
    for p in active_pieces:
        # Shape: 180 dim, Color: 12 dim
        # Gewichte: Form 70%, Farbe 30%
        combined = np.concatenate([
            p['features']['shape'] * 3.5,
            p['features']['color'] * 1.5
        ])
        features.append(combined)

    X = np.array(features)

    # Standardisierung
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchisches Clustering (besser für Puzzles)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='ward',  # Ward für kompakte Cluster
        metric='euclidean'
    )

    labels = clustering.fit_predict(X_scaled)

    return labels


def get_cluster_color(cluster_id):
    """Generiert Farbe für Cluster"""
    if cluster_id == -1:
        return "rgb(180, 180, 180)"

    # Mehr Farben für mehr Cluster
    hue = int((cluster_id * 37) % 180)  # Prime number für bessere Verteilung
    color_hsv = np.uint8([[[hue, 200, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"


# ============================================================================
# CLUSTER-MANAGEMENT
# ============================================================================
def save_cluster_to_db(pieces, cluster_id, cluster_name, db):
    """Speichert Cluster in DB"""
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
    st.set_page_config(page_title="Puzzle Master v3.1", layout="wide")

    if 'pieces' not in st.session_state:
        st.session_state.pieces = []
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}

    db = load_database()

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("⚙️ Steuerung")

        files = st.file_uploader(
            "Bilder hochladen",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )

        st.markdown("---")
        st.subheader("🔧 Parameter")

        min_area = st.slider(
            "Min. Teilgröße (px²)",
            50, 1000, 200, 10,
            help="Höher = weniger kleine Artefakte"
        )

        cluster_dist = st.slider(
            "Cluster-Distanz",
            0.1, 2.0, 0.5, 0.05,
            help="Niedriger = strengere Cluster (0.3-0.7 empfohlen)"
        )

        st.markdown("---")

        if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
            if not files:
                st.warning("Bitte Bilder hochladen!")
            else:
                with st.spinner("Verarbeite..."):
                    all_found = []
                    progress = st.progress(0)

                    for i, f in enumerate(files):
                        img = cv2.imdecode(
                            np.asarray(bytearray(f.read()), dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        all_found.extend(segment_pieces_robust(img, min_area))
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
                    st.success(f"✓ {len(valid)} Teile erkannt!")
                    st.rerun()

        if st.button("🗑️ Session Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("💾 Datenbank")
        st.metric("Teile gesamt", len(db['pieces']))
        st.metric("Cluster", len(db.get('clusters', {})))

    # ========== HAUPTFENSTER ==========
    if st.session_state.pieces:
        # Clustering mit verbessertem Algorithmus
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

            # Re-Cluster Button
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
                st.info("Keine Cluster. Reduziere die Cluster-Distanz oder prüfe ob Teile erkannt wurden.")
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
                st.info("👈 Wähle ein Teil")

        with tab4:
            st.header("💾 Datenbank-Browser")

            db_tab1, db_tab2 = st.tabs(["📋 Alle Teile", "📦 Cluster"])

            with db_tab1:
                st.subheader(f"Alle Teile ({len(db['pieces'])})")

                if not db['pieces']:
                    st.info("DB leer")
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
                                f"📦 {cluster_data['name']} ({cluster_data['piece_count']} Teile)",
                                expanded=False
                        ):
                            st.write(f"**Datum:** {cluster_data['created'][:10]}")
                            st.write(f"**Teile:** {cluster_data['piece_count']}")

                            preview_cols = st.columns(6)
                            for i, p in enumerate(cluster_data['pieces'][:6]):
                                with preview_cols[i]:
                                    st.image(p['thumbnail'])

                            if cluster_data['piece_count'] > 6:
                                st.caption(f"... +{cluster_data['piece_count'] - 6}")

        with tab5:
            st.header("Teile verwalten")

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
                        if st.button(f"❌", key=f"del_{p['id']}", use_container_width=True):
                            p['deleted'] = True
                            st.rerun()

    else:
        st.info("👈 Bilder hochladen")

        st.markdown("---")
        st.markdown("### 🎯 Verbesserte Erkennung")
        st.markdown("""
        **Neue Features:**
        - ✅ Intelligente Filterung weißer/einfarbiger Flächen
        - ✅ Multi-Pass Threshold (3 Methoden kombiniert)
        - ✅ Farb-Varianz-Check (erkennt echte Puzzleteile vs. Artefakte)
        - ✅ Hierarchisches Clustering (stabiler als DBSCAN)
        - ✅ Verbesserte Feature-Kombination (Form + Farbe)

        **Empfohlene Einstellungen:**
        - Min. Teilgröße: 200-400 px²
        - Cluster-Distanz: 0.3-0.7 (niedriger = strenger)
        """)


if __name__ == "__main__":
    main()
