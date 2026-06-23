"""
Sketch-Bot GUI – Web Interface (Gradio)
Starte mit: python3 gui.py
Öffnet sich automatisch im Browser unter http://localhost:7860
"""

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from sketch_generator import process_image, save_sketch


def run_sketch(
    image,
    method,
    canny_weight,
    dodge_weight,
    morph_weight,
    canny_low,
    canny_high,
    blur_sigma,
    clahe_clip,
    clahe_grid,
    final_threshold,
    output_size,
    invert_output,
    background_removal,
):
    if image is None:
        return None, None, "⚠️ Bitte zuerst ein Bild hochladen."
    
    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    
    result, preview = process_image(
        pil_img,
        method=method,
        canny_weight=float(canny_weight),
        dodge_weight=float(dodge_weight),
        morph_weight=float(morph_weight),
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        blur_sigma=float(blur_sigma),
        clahe_clip=float(clahe_clip),
        clahe_grid=int(clahe_grid),
        final_threshold=int(final_threshold),
        output_size=int(output_size),
        invert_output=bool(invert_output),
        background_removal=bool(background_removal),
    )
    
    save_path = "/tmp/portrait_sketch.png"
    save_sketch(result, save_path)
    
    info = (
        f"Methode: **{method}** | "
        f"Ausgabegröße: {result.size[0]}×{result.size[1]}px | "
        f"Gespeichert: `{save_path}`"
    )
    return result, preview, info


def build_ui():
    with gr.Blocks(title="Sketch-Bot Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Photo-to-Sketch Generator
        **Hybrid OpenCV-Pipeline** für den UR5/UFactory Roboter-Input.
        Lade ein Portrait-Foto hoch und erzeuge eine saubere Linienzeichnung.
        """)

        with gr.Row():
            # ── Linke Spalte: Input + Controls ──
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Portrait hochladen",
                    type="numpy",
                    height=300,
                )

                method = gr.Radio(
                    choices=["Hybrid", "Canny (scharf)", "Pencil Dodge (weich)",
                             "Adaptiv (robust)", "Morphologisch"],
                    value="Hybrid",
                    label="Methode",
                )

                with gr.Accordion("⚙️ Hybrid-Gewichtung", open=True):
                    gr.Markdown("*Nur relevant für Hybrid-Methode. Summe sollte ~1.0 sein.*")
                    canny_weight = gr.Slider(0, 1, value=0.5, step=0.05, label="Canny-Gewicht (scharf)")
                    dodge_weight = gr.Slider(0, 1, value=0.3, step=0.05, label="Dodge-Gewicht (weich)")
                    morph_weight = gr.Slider(0, 1, value=0.2, step=0.05, label="Morph-Gewicht (Konturen)")

                with gr.Accordion("🔧 Kantenerkennung", open=False):
                    canny_low  = gr.Slider(10, 150, value=30,  step=5,  label="Canny Low Threshold")
                    canny_high = gr.Slider(50, 300, value=100, step=10, label="Canny High Threshold")
                    blur_sigma = gr.Slider(5, 51,   value=21,  step=2,  label="Dodge Blur Sigma")

                with gr.Accordion("Bildverarbeitung", open=False):
                    clahe_clip  = gr.Slider(0.5, 8.0, value=2.0, step=0.5, label="CLAHE Clip Limit")
                    clahe_grid  = gr.Slider(2, 16,   value=8,   step=2,   label="CLAHE Grid Size")
                    final_threshold = gr.Slider(150, 254, value=230, step=5, label="Finales Threshold")
                    output_size = gr.Slider(256, 2048, value=1024, step=128, label="Ausgabegröße (px)")

                with gr.Accordion("Extras", open=False):
                    invert_output       = gr.Checkbox(value=False,  label="Output invertieren (weiß auf schwarz)")
                    background_removal  = gr.Checkbox(value=True,   label="Hintergrund normieren")

                btn = gr.Button("Skizze erzeugen", variant="primary", size="lg")

            # ── Rechte Spalte: Output ──
            with gr.Column(scale=1):
                result_image  = gr.Image(label="Linienzeichnung (Roboter-Input)", type="pil")
                preview_image = gr.Image(label="Vorschau: Original | Skizze", type="pil")
                status_text   = gr.Markdown("*Noch kein Bild verarbeitet.*")

                gr.Markdown("""
                ---
                ### Verwendung als Roboter-Input
                Die erzeugte PNG wird automatisch als `portrait_sketch.png` gespeichert –
                direkt kompatibel mit dem Sketch-Bot Pipeline-Input (Module 3: Robotic Arm Control).
                
                **Empfohlene Einstellungen je nach Anwendungsfall:**
                | Zweck | Methode | Tipps |
                |---|---|---|
                | Saubere Roboterpfade | Canny (scharf) | Low=20, High=80 |
                | Natürlicher Bleistift-Look | Pencil Dodge | Sigma=25–35 |
                | Ungleichmäßige Beleuchtung | Adaptiv (robust) | — |
                | Beste Gesamtqualität | **Hybrid** | Gewichte anpassen |
                """)

        btn.click(
            fn=run_sketch,
            inputs=[
                image_input, method,
                canny_weight, dodge_weight, morph_weight,
                canny_low, canny_high, blur_sigma,
                clahe_clip, clahe_grid, final_threshold,
                output_size, invert_output, background_removal,
            ],
            outputs=[result_image, preview_image, status_text],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
