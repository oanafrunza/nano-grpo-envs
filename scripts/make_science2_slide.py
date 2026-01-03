#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont

# Default input paths
BASE = os.path.join(os.getcwd(), "exp_output", "science2_suite", "_analysis")
LEFT_PATH = os.path.join(BASE, "overall_pass_at1.png")
RIGHT_PATH = os.path.join(BASE, "per_problem_delta_vs_baseline.png")
OUT_PATH = os.path.join(BASE, "science2_main_suite_slide.png")

TITLE = "Science2 Main Suite"
LEFT_LABEL = "Overall Pass@1"
RIGHT_LABEL = "Per-Problem Δ vs Baseline"

PADDING = 24
TOP_PAD = 60
BOTTOM_PAD = 24
LABEL_PAD = 32
BG_COLOR = (248, 249, 250)
TEXT_COLOR = (33, 37, 41)


def load_img(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input image: {path}")
    return Image.open(path).convert("RGB")


def draw_title(draw: ImageDraw.ImageDraw, canvas_w: int, title: str, font: ImageFont.ImageFont):
    bbox = draw.textbbox((0, 0), title, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((canvas_w - w) // 2, PADDING), title, fill=TEXT_COLOR, font=font)


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, font: ImageFont.ImageFont):
    draw.text((x, y), label, fill=TEXT_COLOR, font=font)


def main():
    # Allow overriding via env vars if desired
    left_path = os.environ.get("SCIENCE2_SLIDE_LEFT", LEFT_PATH)
    right_path = os.environ.get("SCIENCE2_SLIDE_RIGHT", RIGHT_PATH)
    out_path = os.environ.get("SCIENCE2_SLIDE_OUT", OUT_PATH)

    left = load_img(left_path)
    right = load_img(right_path)

    # Normalize heights to the same target while preserving aspect
    target_inner_height = 800
    def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
        w, h = img.size
        new_w = int(w * (target_h / h))
        return img.resize((new_w, target_h), Image.LANCZOS)

    left_r = resize_to_height(left, target_inner_height)
    right_r = resize_to_height(right, target_inner_height)

    # Layout calculations
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 36)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Measure labels to add extra vertical space
    dummy = Image.new("RGB", (1, 1))
    ddraw = ImageDraw.Draw(dummy)
    def text_h(draw_obj, txt, font):
        bbox = draw_obj.textbbox((0, 0), txt, font=font)
        return bbox[3] - bbox[1]
    title_h = text_h(ddraw, TITLE, title_font)
    left_lab_h = text_h(ddraw, LEFT_LABEL, label_font)
    right_lab_h = text_h(ddraw, RIGHT_LABEL, label_font)
    label_h = max(left_lab_h, right_lab_h)

    canvas_w = PADDING + left_r.size[0] + PADDING + right_r.size[0] + PADDING
    canvas_h = PADDING + title_h + TOP_PAD + label_h + LABEL_PAD + target_inner_height + BOTTOM_PAD

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Title centered
    draw_title(draw, canvas_w, TITLE, title_font)

    # Labels above each image
    left_x = PADDING
    right_x = PADDING + left_r.size[0] + PADDING
    labels_y = PADDING + title_h + TOP_PAD
    draw_label(draw, left_x, labels_y, LEFT_LABEL, label_font)
    draw_label(draw, right_x, labels_y, RIGHT_LABEL, label_font)

    # Images
    imgs_y = labels_y + label_h + LABEL_PAD
    canvas.paste(left_r, (left_x, imgs_y))
    canvas.paste(right_r, (right_x, imgs_y))

    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote slide: {out_path}")


if __name__ == "__main__":
    main()
