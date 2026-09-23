#!/usr/bin/env python3
"""Document scanner — clean PDF from paper photos.

Detects the paper sheet in each photo as the largest 4-corner contour, applies
perspective transform to flatten it, optionally splits open-book spreads at
the binding, and combines the pages into a single A4 PDF.

Use this when you've photographed pages of a book or notebook with a phone
(flat from above, paper on contrasting background) and want a clean scan
without re-shooting with the iPhone Notes scanner.

Usage:
    scan.py [INPUT_DIR] [-o OUTPUT_PDF] [--split-spreads] [--cover N ...]

Examples:
    # Default: all JPGs in cwd, output ./scanned.pdf, no splitting.
    scan.py

    # Folder of spread photos, split every photo at the binding.
    scan.py ~/Downloads/book/ -o ~/Downloads/book.pdf --split-spreads

    # Mixed: first photo is a single-page cover, the rest are spreads.
    scan.py ~/Downloads/book/ --split-spreads --cover 0

Requires: opencv-python, numpy, pillow.
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


# --- core image ops ----------------------------------------------------------

def load_oriented(path: Path) -> np.ndarray:
    """Read with PIL (auto-orient via EXIF), return BGR ndarray."""
    pil = Image.open(path)
    pil = pil.convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return [TL, TR, BR, BL] given an arbitrary 4-point array."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def find_paper_corners(bgr: np.ndarray) -> np.ndarray | None:
    """Find the 4 corners of the largest paper-sized quad in the image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    min_area = 0.25 * bgr.shape[0] * bgr.shape[1]
    candidates: list[np.ndarray] = []

    # Edge-based strategy
    edged = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            candidates.append(approx)
            break

    # Brightness-threshold strategy
    _, th = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            candidates.append(approx)
            break
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) > 0.30 * bgr.shape[0] * bgr.shape[1]:
            candidates.append(box.reshape(4, 1, 2).astype(np.int32))
            break

    if not candidates:
        return None
    return order_corners(max(candidates, key=cv2.contourArea))


def warp_to_rect(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply perspective transform so the quad fills a clean rectangle."""
    tl, tr, br, bl = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([
        [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


def gentle_cleanup(bgr: np.ndarray) -> np.ndarray:
    """Light contrast bump preserving original feel of the paper.

    Aggressive whitening washes out faint pencil. This stretches blacks-50 to 255
    so pencil stays readable and paper goes near-white without losing texture.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    norm = np.clip((gray.astype(np.float32) - 50) * (255.0 / 175.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def split_at_binding(warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the darkest vertical band near horizontal centre and split there."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    band_start, band_end = int(w * 0.42), int(w * 0.58)
    col_darkness = (255 - gray[h // 4: 3 * h // 4, band_start:band_end]).sum(axis=0)
    best = band_start + int(np.argmax(col_darkness))
    return warped[:, :best], warped[:, best:]


def save_a4(bgr: np.ndarray, dst: Path, target_w: int = 2480, target_h: int = 3508) -> None:
    """Resize to fit within A4 portrait and pad with white background."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    iw, ih = pil.size
    if iw > ih:
        pil = pil.rotate(90, expand=True)
        iw, ih = pil.size
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), "white")
    canvas.paste(pil, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    canvas.save(dst, "JPEG", quality=92)


# --- pipeline ----------------------------------------------------------------

def process_photo(path: Path, work_dir: Path, pages_dir: Path,
                  base_index: int, is_spread: bool) -> int:
    """Process one photo. Returns the number of pages written."""
    bgr = load_oriented(path)
    corners = find_paper_corners(bgr)
    if corners is None:
        warped = bgr
        print(f"  WARN {path.name}: paper corners not detected, using whole frame")
    else:
        warped = warp_to_rect(bgr, corners)

    # cv2.imwrite can't open non-ASCII paths on Windows; encode in memory instead.
    (work_dir / f"{path.stem}-warped.jpg").write_bytes(cv2.imencode(".jpg", warped)[1].tobytes())

    if is_spread:
        left, right = split_at_binding(warped)
        save_a4(gentle_cleanup(left),  pages_dir / f"{base_index:03d}.jpg")
        save_a4(gentle_cleanup(right), pages_dir / f"{base_index + 1:03d}.jpg")
        return 2
    save_a4(gentle_cleanup(warped), pages_dir / f"{base_index:03d}.jpg")
    return 1


def build_pdf(input_dir: Path, output_pdf: Path,
              split_spreads: bool, cover_indices: set[int]) -> None:
    work_dir  = output_pdf.with_suffix("").with_name(output_pdf.stem + ".work")
    pages_dir = output_pdf.with_suffix("").with_name(output_pdf.stem + ".pages")
    if work_dir.exists():  shutil.rmtree(work_dir)
    if pages_dir.exists(): shutil.rmtree(pages_dir)
    work_dir.mkdir(parents=True)
    pages_dir.mkdir(parents=True)

    photos = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))
    if not photos:
        sys.exit(f"No image files in {input_dir}")

    next_index = 1
    for i, p in enumerate(photos):
        is_spread = split_spreads and i not in cover_indices
        print(f"Processing {p.name} ({'spread' if is_spread else 'single page'})")
        next_index += process_photo(p, work_dir, pages_dir, next_index, is_spread)

    pages = sorted(pages_dir.glob("*.jpg"))
    imgs = [Image.open(p).convert("RGB") for p in pages]
    imgs[0].save(output_pdf, "PDF", resolution=300, save_all=True, append_images=imgs[1:])
    print(f"\nPDF: {output_pdf} ({output_pdf.stat().st_size / 1024:.0f} KB, {len(pages)} pages)")


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    # A Windows console or pipe may not be UTF-8; print what it can rather than crash.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", nargs="?", default=".",
                    help="Folder of photos (default: current directory)")
    ap.add_argument("-o", "--output", default="scanned.pdf",
                    help="Output PDF path (default: ./scanned.pdf)")
    ap.add_argument("--split-spreads", action="store_true",
                    help="Each photo is a 2-page spread; split at the binding")
    ap.add_argument("--cover", type=int, action="append", default=[],
                    metavar="INDEX",
                    help="Photo INDEX (0-based) is a single page, not a spread. Repeatable.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_pdf = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(input_dir, output_pdf, args.split_spreads, set(args.cover))


if __name__ == "__main__":
    main()
