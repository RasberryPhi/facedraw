"""
Sketch-Bot CLI – Roboter-optimierte Version
Verwendung:
    python3 cli.py --input portrait.jpg --output portrait_sketch.png --preview

Zonen-Parameter:
    --eye-top       Obere Grenze Augenzone  (Standard: 0.30 = 30% Bildhöhe)
    --eye-bottom    Untere Grenze Augenzone (Standard: 0.55)
    --shirt-top     Beginn Kleiderzone      (Standard: 0.70)

Linienlängen pro Zone (kleiner = mehr Linien):
    --arc-eye       Augen   (Standard: 6)
    --arc-face      Gesicht (Standard: 15)
    --arc-shirt     Kleidung (Standard: 150)
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from sketch_generator import process_image, save_sketch


def main():
    p = argparse.ArgumentParser(
        description="Photo-to-Sketch Generator für UR5 Roboter-Input",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Ein/Ausgabe
    p.add_argument("--input",  "-i", required=True,                  help="Eingabebild (JPG/PNG)")
    p.add_argument("--output", "-o", default="portrait_sketch.png",  help="Ausgabe-PNG")
    p.add_argument("--preview",      action="store_true",            help="Seite-an-Seite Vorschau speichern")

    # Kantenerkennung
    p.add_argument("--canny-low",    type=int,   default=12,   help="Canny untere Schwelle  (weniger = mehr Linien)")
    p.add_argument("--canny-high",   type=int,   default=65,   help="Canny obere Schwelle")

    # Glättung
    p.add_argument("--bilateral-d",  type=int,   default=9,    help="Bilateral Filter Stärke")
    p.add_argument("--bilateral-col",type=int,   default=60,   help="Bilateral Farb-Sigma")

    # Zonen-Grenzen
    p.add_argument("--eye-top",      type=float, default=0.30, help="Augenzone oben   (0.0–1.0)")
    p.add_argument("--eye-bottom",   type=float, default=0.55, help="Augenzone unten  (0.0–1.0)")
    p.add_argument("--shirt-top",    type=float, default=0.70, help="Kleiderzone ab   (0.0–1.0)")

    # Mindestlängen pro Zone
    p.add_argument("--arc-eye",      type=float, default=6,    help="Mindestlänge Augen    (px)")
    p.add_argument("--arc-face",     type=float, default=15,   help="Mindestlänge Gesicht  (px)")
    p.add_argument("--arc-shirt",    type=float, default=150,  help="Mindestlänge Kleidung (px)")

    # Maximalflächen pro Zone
    p.add_argument("--area-eye",     type=float, default=5000, help="Maxfläche Augen    (px²)")
    p.add_argument("--area-face",    type=float, default=8000, help="Maxfläche Gesicht  (px²)")
    p.add_argument("--area-shirt",   type=float, default=2000, help="Maxfläche Kleidung (px²)")

    # Sonstiges
    p.add_argument("--thickness",    type=int,   default=2,    help="Liniendicke in px")
    p.add_argument("--size",         type=int,   default=1024, help="Ausgabegröße in px")
    p.add_argument("--no-bg-norm",   action="store_true",      help="Hintergrundnormierung deaktivieren")

    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"❌ Datei nicht gefunden: {args.input}")
        sys.exit(1)

    print(f"📷 Lade: {path}")
    img = Image.open(path)
    print(f"🎨 Methode: Roboter-Portrait (Zonen-Filter)")
    print(f"   Zonen: Augen={args.eye_top}–{args.eye_bottom} | Kleidung ab {args.shirt_top}")
    print(f"   Mindestlängen: Augen={args.arc_eye}px | Gesicht={args.arc_face}px | Kleidung={args.arc_shirt}px")

    result, preview = process_image(
        img,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        bilateral_d=args.bilateral_d,
        bilateral_color=args.bilateral_col,
        eye_top=args.eye_top,
        eye_bottom=args.eye_bottom,
        shirt_top=args.shirt_top,
        arc_eye=args.arc_eye,
        arc_face=args.arc_face,
        arc_shirt=args.arc_shirt,
        area_eye=args.area_eye,
        area_face=args.area_face,
        area_shirt=args.area_shirt,
        line_thickness=args.thickness,
        output_size=args.size,
        background_removal=not args.no_bg_norm,
    )

    save_sketch(result, args.output)
    print(f"✅ Skizze gespeichert: {args.output} ({result.size[0]}×{result.size[1]}px)")

    if args.preview:
        prev_path = Path(args.output).stem + "_preview.png"
        preview.save(prev_path)
        print(f"👁️  Vorschau gespeichert: {prev_path}")


if __name__ == "__main__":
    main()
