# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthesize an image-edit dataset (images/ + train.jsonl + test.jsonl) for
Qwen-Image-Edit RL training.

Writes exactly the layout that prepare_data.py in this directory consumes, so
the two scripts chain into the train.parquet / test.parquet files expected by
run_qwen_image_edit_lora.sh. Every instruction is grounded in the rendered
scene's actual attributes (colors, shapes, text), which keeps reward signals
like PickScore meaningful. No network access and no GPU needed.

Example:
    python generate_data.py --train_size 1024 --val_size 64
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PALETTE = {
    "red": (214, 69, 65),
    "blue": (68, 108, 179),
    "green": (46, 134, 83),
    "yellow": (241, 196, 15),
    "purple": (142, 68, 173),
    "orange": (230, 126, 34),
    "pink": (231, 130, 172),
    "brown": (121, 85, 61),
    "teal": (22, 133, 126),
    "gray": (127, 140, 141),
}
SHAPES = ("circle", "square", "triangle")
PLURAL = {"circle": "circles", "square": "squares", "triangle": "triangles"}
NUM_WORDS = {1: "one", 2: "two", 3: "three"}
WORDS = ("hello", "dream", "sunny", "orbit", "meadow", "signal")
GENERIC_EDITS = (
    "Make the image brighter.",
    "Make the image darker.",
    "Blur the whole image slightly.",
    "Increase the contrast of the image.",
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _draw_background(rng: random.Random, size: int) -> tuple[Image.Image, str]:
    # composite takes image1 where the mask is white (bottom), so `first` lands
    # at the bottom of the canvas and `second` at the top
    (first_name, first_rgb), (second_name, second_rgb) = rng.sample(list(PALETTE.items()), 2)
    gradient = Image.linear_gradient("L").resize((size, size))
    canvas = Image.composite(
        Image.new("RGB", (size, size), first_rgb),
        Image.new("RGB", (size, size), second_rgb),
        gradient,
    )
    if rng.random() < 0.5:  # flatten half the scenes to a solid color
        canvas = Image.new("RGB", (size, size), second_rgb)
    return canvas, second_name


def _draw_shapes(rng: random.Random, draw: ImageDraw.ImageDraw, size: int) -> list[dict]:
    shapes = []
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(SHAPES)
        name, rgb = rng.choice(list(PALETTE.items()))
        w, h = rng.randint(size // 8, size // 3), rng.randint(size // 8, size // 3)
        if kind == "circle":
            h = w  # keep it a circle, instructions call it one
        x0, y0 = rng.randint(0, size - w), rng.randint(0, size - h)
        box = (x0, y0, x0 + w, y0 + h)
        if kind == "circle":
            draw.ellipse(box, fill=rgb)
        elif kind == "square":
            draw.rectangle(box, fill=rgb)
        else:
            draw.polygon([(x0, y0 + h), (x0 + w, y0 + h), (x0 + w // 2, y0)], fill=rgb)
        shapes.append({"kind": kind, "color": name})
    return shapes


def _draw_text(rng: random.Random, draw: ImageDraw.ImageDraw, size: int) -> str:
    word = rng.choice(WORDS)
    font = _font(size // 8)
    draw.text((size // 20, size // 20), word, fill=(255, 255, 255), font=font)
    return word


def _render_scene(rng: random.Random, size: int) -> tuple[Image.Image, dict]:
    canvas, bg_color = _draw_background(rng, size)
    draw = ImageDraw.Draw(canvas)
    scene = {"bg_color": bg_color, "shapes": [], "text": None}
    if rng.random() < 0.75:
        scene["shapes"] = _draw_shapes(rng, draw, size)
    if rng.random() < 0.5:
        scene["text"] = _draw_text(rng, draw, size)
    return canvas, scene


def _make_instruction(rng: random.Random, scene: dict) -> str:
    bg, shapes, text = scene["bg_color"], scene["shapes"], scene["text"]
    families = ["bg", "generic"]
    if shapes:
        families += ["add", "remove", "recolor"]
    if text:
        families += ["text_change", "text_remove"]
    else:
        families += ["text_add"]
    family = rng.choice(families)

    if family == "bg":
        target = rng.choice([c for c in PALETTE if c != bg])
        return f"Change the background color to {target}."
    if family == "add":
        count = rng.choice(list(NUM_WORDS))
        kind = rng.choice(SHAPES)
        color = rng.choice(list(PALETTE))
        noun = PLURAL[kind] if count > 1 else kind
        return f"Add {NUM_WORDS[count]} {color} {noun} to the image."
    if family == "remove":
        shape = rng.choice(shapes)
        return f"Remove the {shape['color']} {shape['kind']}."
    if family == "recolor":
        shape = rng.choice(shapes)
        target = rng.choice([c for c in PALETTE if c != shape["color"]])
        return f"Change the {shape['kind']} from {shape['color']} to {target}."
    if family == "text_change":
        return f"Change the text '{text}' to '{rng.choice(WORDS)}'."
    if family == "text_remove":
        return f"Remove the text '{text}' from the image."
    if family == "text_add":
        return f"Add the text '{rng.choice(WORDS)}' at the top of the image."
    if family == "generic":
        # instruction-only dataset: the rollout model applies edits, this script never does
        return rng.choice(GENERIC_EDITS)
    raise ValueError(f"unknown instruction family: {family}")


def _write_split(rng: random.Random, out_dir: Path, split: str, count: int, size: int) -> None:
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as sink:
        for index in range(count):
            canvas, scene = _render_scene(rng, size)
            instruction = _make_instruction(rng, scene)
            name = f"{split}_{index:06d}.png"
            canvas.save(image_dir / name, format="PNG")
            sink.write(json.dumps({"prompt": instruction, "image": name}) + "\n")


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_size", type=int, default=1024)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512, help="square side; must match rollout height/width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw_dir", type=Path, default=repo_dir / "data" / "qwen_image_edit_raw")
    parser.add_argument("--parquet_dir", type=Path, default=repo_dir / "data" / "qwen_image_edit")
    parser.add_argument("--no_parquet", action="store_true", help="write only images/ + jsonl, skip prepare_data.py")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    _write_split(rng, args.raw_dir, "train", args.train_size, args.image_size)
    _write_split(rng, args.raw_dir, "test", args.val_size, args.image_size)
    print(f"Wrote {args.train_size} train / {args.val_size} test samples to {args.raw_dir}")

    if not args.no_parquet:
        converter = Path(__file__).resolve().parent / "prepare_data.py"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--input_dir",
                str(args.raw_dir),
                "--output_dir",
                str(args.parquet_dir),
                "--image_size",
                str(args.image_size),
            ],
            check=True,
        )

    # self-check: every jsonl line has its image and a non-empty instruction
    for split, expected in (("train", args.train_size), ("test", args.val_size)):
        lines = (args.raw_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected, f"{split}: {len(lines)} lines, expected {expected}"
        for line in lines:
            example = json.loads(line)
            assert example["prompt"] and (args.raw_dir / "images" / example["image"]).is_file(), example
    print("Self-check passed: jsonl line counts match and every condition image exists.")


if __name__ == "__main__":
    main()
