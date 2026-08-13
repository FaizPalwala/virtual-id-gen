#!/usr/bin/env python3
"""Compose a dissertation-quality 3×3 hyperparameter grid from the sweep.

Layout (academic matrix):
    rows    = guidance scale   (3.0 / 4.0 / 5.5)
    columns = IP-adapter scale (0.75 / 0.85 / 0.90)
    each cell = the 3 candidate images edge-to-edge, each with a small
                corner badge showing its actual generation seed (read
                from the combo's raw_candidate_manifest.csv)

The default combo (3.0, 0.85) — the config used for the main dataset —
gets a green border and green labels.  Reads ``sweep_juggernaut/<gs>_<ips>/
candidates/identity_000/candidate_*.png``.  The grid IS the artifact:
``--cleanup`` deletes the source candidate PNGs once the grid is written.

Usage:
    python scripts/make_sweep_grid.py --sweep-root <dir> [--cleanup]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GUIDANCE_ORDER = ["3.0", "4.0", "5.5"]
IP_ADAPTER_ORDER = ["0.75", "0.85", "0.90"]
N_SEEDS = 3

# Highlight the pipeline default combo (gs=3.0, ips=0.85) — green border
# + green label (matches the guidance green already in the palette).
DEFAULT_COMBO = ("3.0", "0.85")
DEFAULT_MARK = "#50C878"

# ── Canvas metrics — tuned for A4 print legibility.  The figure is ~1900 px
# wide ≈ 6 in at ~320 DPI at text width; fonts are sized as fractions of
# the image cell.
IMG = 180          # px per candidate image inside a cell
SEED_GAP = 0       # images edge-to-edge inside a cell
SEED_BADGE = 26    # in-cell seed badge (small square, image corner)
CELL_W = 3 * IMG + 2 * SEED_GAP          # 540
CELL_H = IMG                              # 180
AXIS_H = 30        # rotated axis-title band height ("IP-adapter scale")
VALUE_H = 44       # column-value band
LABEL_HEIGHT = AXIS_H + VALUE_H          # 74
ROW_LABEL_W = 170  # left margin: rotated axis title + row values
PADDING = 12       # gap between cells
BG = (30, 30, 30)
FG = (235, 235, 235)
FG_DIM = (170, 170, 170)
FONT_AXIS = 26
FONT_VALUE = 30
FONT_SEED = 15


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _combos(sweep_root: Path) -> list[tuple[str, str]]:
    """All present (guidance, ip_adapter) combos in display order."""
    combos = []
    for gs in GUIDANCE_ORDER:
        for ips in IP_ADAPTER_ORDER:
            if (sweep_root / f"{gs}_{ips}").is_dir():
                combos.append((gs, ips))
    return combos


def _load_cell(combo_dir: Path) -> list[Image.Image]:
    """Load the 3 candidates (RGB) for a combo, in seed order."""
    imgs = []
    for idx in range(N_SEEDS):
        p = combo_dir / "candidates" / "identity_000" / f"candidate_{idx:03d}.png"
        if not p.is_file():
            raise FileNotFoundError(f"missing sweep candidate: {p}")
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


def _generation_seeds(combo_dir: Path) -> list[str]:
    """Actual generation seeds (trial 0..2) from the combo's manifest."""
    man = combo_dir / "raw_candidate_manifest.csv"
    seeds = []
    if man.is_file():
        with man.open(newline="") as f:
            for row in csv.DictReader(f):
                seeds.append(str(row.get("generationseed", "")))
    return seeds[:N_SEEDS]


def _cleanup_combo(combo_dir: Path) -> None:
    """Delete the consumed candidate PNGs for one combo."""
    cand_dir = combo_dir / "candidates" / "identity_000"
    n = 0
    for p in sorted(cand_dir.glob("candidate_*.png")):
        p.unlink()
        n += 1
    for d in (cand_dir, cand_dir.parent):
        try:
            d.rmdir()
        except OSError:
            pass
    print(f"  cleaned {n} candidates from {combo_dir.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a dissertation-quality 3×3 hyperparameter grid "
                    "from the Juggernaut sweep outputs"
    )
    parser.add_argument(
        "--sweep-root", required=True,
        help="Directory containing <gs>_<ips>/ combo subdirs",
    )
    parser.add_argument(
        "--out", default="sweep_grid.png",
        help="Output PNG path (default: sweep_grid.png)",
    )
    parser.add_argument(
        "--image-size", type=int, default=IMG,
        help="Display size in px of each 1024px candidate (default: 180)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete source candidate PNGs after the grid is written",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    if not sweep_root.is_dir():
        print(f"FAIL: sweep root not found: {sweep_root}", file=sys.stderr)
        return 1

    combos = _combos(sweep_root)
    if len(combos) != 9:
        print(f"FAIL: expected 9 combos, found {len(combos)}: {combos}",
              file=sys.stderr)
        return 1

    # Load everything up front — fail fast on any gap, nothing partial.
    cells = {}
    for combo in combos:
        gs, ips = combo
        cells[combo] = _load_cell(sweep_root / f"{gs}_{ips}")
    seeds = _generation_seeds(sweep_root / f"{combos[0][0]}_{combos[0][1]}")

    # Resampling filter (Pillow >= 9.1: Image.Resampling.LANCZOS).
    from PIL import Image as _Image
    try:
        _LANCZOS = _Image.Resampling.LANCZOS
    except AttributeError:  # pragma: no cover — old Pillow fallback
        _LANCZOS = _Image.LANCZOS

    img = args.image_size
    cell_w = 3 * img + 2 * SEED_GAP
    cell_h = img

    # Consistent padding: PADDING above the label band, between bands and
    # cells, and at the bottom — same rhythm as the cell gaps.
    label_y0 = PADDING                    # top of the axis+value bands
    grid_x0 = ROW_LABEL_W + PADDING       # left edge of the cell grid
    grid_y0 = label_y0 + LABEL_HEIGHT + PADDING

    canvas_w = grid_x0 + 3 * cell_w + 2 * PADDING + PADDING
    canvas_h = grid_y0 + 3 * cell_h + 2 * PADDING + PADDING
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    font_axis = _font(FONT_AXIS)
    font_value = _font(FONT_VALUE)
    font_seed = _font(FONT_SEED)

    # ── Column headers: axis title (centred over the 3 columns) + values ──
    title = "IP-adapter scale"
    tw = draw.textbbox((0, 0), title, font=font_axis)[2]
    grid_w = 3 * cell_w + 2 * PADDING
    draw.text((grid_x0 + (grid_w - tw) // 2,
               label_y0 + (AXIS_H - FONT_AXIS) // 2),
              title, fill=FG_DIM, font=font_axis)
    for ii, ips in enumerate(IP_ADAPTER_ORDER):
        x0 = grid_x0 + ii * (cell_w + PADDING)
        is_default = (DEFAULT_COMBO[1] == ips)
        val_fg = DEFAULT_MARK if is_default else FG
        val = ips
        tw = draw.textbbox((0, 0), val, font=font_value)[2]
        draw.text((x0 + (cell_w - tw) // 2,
                   label_y0 + AXIS_H + (VALUE_H - FONT_VALUE) // 2),
                  val, fill=val_fg, font=font_value)

    # ── Row headers: guidance value per row (no colour bar, no "=") ──
    for gi, gs in enumerate(GUIDANCE_ORDER):
        y0 = grid_y0 + gi * (cell_h + PADDING)
        val = gs
        tw = draw.textbbox((0, 0), val, font=font_value)[2]
        draw.text((ROW_LABEL_W - 14 - tw, y0 + (cell_h - FONT_VALUE) // 2),
                  val, fill=FG, font=font_value)

    # ── Rotated axis title: "Guidance scale" (left margin, vertical) ──
    # Size the pre-rotation canvas to the TEXT bbox, not the grid height —
    # otherwise rotation expands it across the whole grid and clips the
    # label (the "riding up / invisible" bug).
    axis_txt = "Guidance scale"
    bb = draw.textbbox((0, 0), axis_txt, font=font_axis)
    tw_axis = int(bb[2] - bb[0])
    th_axis = int(bb[3] - bb[1])
    axis_img = Image.new("RGB", (tw_axis + 12, th_axis + 12), BG)
    axis_draw = ImageDraw.Draw(axis_img)
    axis_draw.text((6 - bb[0], 6 - bb[1]), axis_txt, fill=FG_DIM,
                   font=font_axis)
    axis_img = axis_img.rotate(90, expand=True)  # (th+12) x (tw+12)
    ax_w, ax_h = axis_img.size
    grid_h = 3 * cell_h + 2 * PADDING
    canvas.paste(axis_img,
                 ((ROW_LABEL_W - ax_w) // 2, grid_y0 + (grid_h - ax_h) // 2))

    # ── Cells: 3 seeds edge-to-edge, seed badge in each image corner ──
    badge_sz = SEED_BADGE
    for gi, gs in enumerate(GUIDANCE_ORDER):
        for ii, ips in enumerate(IP_ADAPTER_ORDER):
            combo = (gs, ips)
            x0 = grid_x0 + ii * (cell_w + PADDING)
            y0 = grid_y0 + gi * (cell_h + PADDING)
            for si, seed_img in enumerate(cells[combo]):
                sx = x0 + si * (img + SEED_GAP)
                resized = (seed_img.resize((img, img), _LANCZOS)
                           if seed_img.size != (img, img) else seed_img)
                canvas.paste(resized, (sx, y0))
                # seed badge: small dark square in the image's top-left
                # corner with the actual generation seed number
                draw.rectangle([sx, y0, sx + badge_sz, y0 + badge_sz],
                               fill=(20, 20, 20))
                marker = seeds[si] if si < len(seeds) else str(si + 1)
                tw = draw.textbbox((0, 0), marker, font=font_seed)[2]
                draw.text((sx + (badge_sz - tw) // 2,
                           y0 + (badge_sz - FONT_SEED) // 2),
                          marker, fill=FG_DIM, font=font_seed)
            if combo == DEFAULT_COMBO:
                draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h],
                               outline=DEFAULT_MARK, width=4)

    out = Path(args.out)
    # Uniform border: crop the canvas to the content bounding box + PADDING
    # on every side, so left/right/top/bottom margins match exactly.
    from PIL import ImageChops

    diff = ImageChops.difference(
        canvas, Image.new("RGB", canvas.size, BG)
    ).convert("L")
    bbox = diff.getbbox()
    if bbox:
        l, t, r, b = bbox
        canvas = canvas.crop((
            max(l - PADDING, 0), max(t - PADDING, 0),
            min(r + PADDING, canvas.width), min(b + PADDING, canvas.height),
        ))

    canvas.save(out)
    print(f"Grid → {out} ({canvas.width}×{canvas.height}, "
          f"{out.stat().st_size / 1e6:.1f} MB)")

    if args.cleanup:
        for gs, ips in combos:
            _cleanup_combo(sweep_root / f"{gs}_{ips}")
        print("Sweep candidates consumed; metadata (manifests) kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
