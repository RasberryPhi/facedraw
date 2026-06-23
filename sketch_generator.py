"""
Photo-to-Sketch Generator – Roboter-optimierte Version
Erzeugt saubere, durchgehende Linien als UR5-Input

Zonen-basierter Filter:
  - Augenbereich (30–55%): sehr empfindlich → Pupillen, Wimpern
  - Gesicht (Rest oben):   normal → Konturen, Ohren, Nase, Mund
  - Kleidung (ab 70%):     minimal → nur Körperaußenkontur
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def _normalize_bg(gray: np.ndarray, sigma: float = 60) -> np.ndarray:
    """Hintergrundbeleuchtung normieren."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return cv2.divide(gray, bg, scale=255)


def robot_portrait(
    gray: np.ndarray,
    canny_low: int = 12,
    canny_high: int = 65,
    bilateral_d: int = 9,
    bilateral_color: int = 60,
    # Zonen-Grenzen (Anteil der Bildhöhe)
    eye_top: float = 0.20,
    eye_bottom: float = 0.55,
    shirt_top: float = 0.70,
    # Mindest-Linienlängen pro Zone
    arc_eye: float = 6,
    arc_face: float = 15,
    arc_shirt: float = 150,
    # Maximale Fläche pro Zone
    area_eye: float = 4000,
    area_face: float = 8000,
    area_shirt: float = 2000,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Roboter-optimierte Portrait-Skizze mit zonenbasiertem Filter.

    Zonen:
      - Augen    (eye_top  – eye_bottom): arc > arc_eye   → Pupillen sichtbar
      - Gesicht  (oben bis shirt_top):    arc > arc_face  → normale Konturen
      - Kleidung (ab shirt_top):          arc > arc_shirt → nur Außenkontur
    """
    h, w = gray.shape

    # 1. Vorverarbeitung
    normalized = _normalize_bg(gray)
    smooth = cv2.bilateralFilter(
        normalized, d=bilateral_d,
        sigmaColor=bilateral_color,
        sigmaSpace=bilateral_color
    )

    # 2. Kantenerkennung
    edges = cv2.Canny(smooth, canny_low, canny_high)

    # 3. Zonengrenzen in Pixel
    eye_top_px    = int(h * eye_top)
    eye_bottom_px = int(h * eye_bottom)
    shirt_top_px  = int(h * shirt_top)

    # 4. Konturfilter zonenbasiert
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.ones_like(edges) * 255  # weißer Hintergrund

    for cnt in contours:
        arc_len = cv2.arcLength(cnt, closed=False)
        area    = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        center_y = y + ch // 2

        if center_y > shirt_top_px:
            # Kleidung: nur lange Außenkonturen
            if arc_len > arc_shirt and area < area_shirt:
                cv2.drawContours(canvas, [cnt], -1, 0, line_thickness)

        elif eye_top_px < center_y < eye_bottom_px:
            # Augenbereich: sehr empfindlich
            if arc_len > arc_eye and area < area_eye:
                cv2.drawContours(canvas, [cnt], -1, 0, line_thickness)

        else:
            # Gesicht (Stirn, Nase, Mund, Ohren, Kinn)
            if arc_len > arc_face and area < area_face:
                cv2.drawContours(canvas, [cnt], -1, 0, line_thickness)

    # 5. Lücken schließen → durchgehende Roboterpfade
    canvas_inv = cv2.bitwise_not(canvas)
    k_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(canvas_inv, cv2.MORPH_CLOSE, k_close)
    result = cv2.bitwise_not(closed)

    return result


def process_image(
    pil_image: Image.Image,
    # Canny
    canny_low: int = 12,
    canny_high: int = 65,
    # Bilateral
    bilateral_d: int = 9,
    bilateral_color: int = 60,
    # Zonen
    eye_top: float = 0.30,
    eye_bottom: float = 0.55,
    shirt_top: float = 0.70,
    # Mindestlängen
    arc_eye: float = 6,
    arc_face: float = 15,
    arc_shirt: float = 150,
    # Maximalflächen
    area_eye: float = 5000,
    area_face: float = 8000,
    area_shirt: float = 2000,
    # Sonstiges
    line_thickness: int = 2,
    output_size: int = 1024,
    background_removal: bool = True,
) -> Tuple[Image.Image, Image.Image]:
    """
    Verarbeitet ein Portrait-Foto zu einer Roboter-optimierten Linienzeichnung.
    Gibt (Skizze, Vorschau_Seite_an_Seite) zurück.
    """
    # PIL → OpenCV
    img_np = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Resize
    h, w = img_bgr.shape[:2]
    scale = output_size / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if background_removal:
        gray = _normalize_bg(gray)

    sketch = robot_portrait(
        gray,
        canny_low=canny_low,
        canny_high=canny_high,
        bilateral_d=bilateral_d,
        bilateral_color=bilateral_color,
        eye_top=eye_top,
        eye_bottom=eye_bottom,
        shirt_top=shirt_top,
        arc_eye=arc_eye,
        arc_face=arc_face,
        arc_shirt=arc_shirt,
        area_eye=area_eye,
        area_face=area_face,
        area_shirt=area_shirt,
        line_thickness=line_thickness,
    )

    new_h, new_w = sketch.shape[:2]
    result_pil = Image.fromarray(sketch)

    # Side-by-Side Vorschau
    orig_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    preview = Image.new("RGB", (new_w * 2 + 20, new_h), (200, 200, 200))
    preview.paste(orig_pil, (0, 0))
    preview.paste(result_pil.convert("RGB"), (new_w + 20, 0))

    return result_pil, preview


def save_sketch(pil_image: Image.Image, output_path: str = "portrait_sketch.png") -> str:
    pil_image.save(output_path)
    return output_path
