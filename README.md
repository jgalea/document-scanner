<h1 align="center">document-scanner</h1>

<p align="center">
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/PYTHON-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-5C9E31?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/OFFLINE-no%20network%20calls-brightgreen?style=for-the-badge" alt="Offline">
  <a href="https://rebelcode.com"><img src="https://img.shields.io/badge/BUILT%20BY-REBELCODE-8A2BE2?style=for-the-badge" alt="Built by RebelCode"></a>
</p>

<p align="center"><strong>Turn phone photos of paper into a clean, straightened PDF. For the pages you already photographed and can't reshoot.</strong></p>

---

## Overview

Point it at a folder of photos. It finds the sheet of paper in each one, flattens it, and writes a single A4 PDF.

If you can go back and rescan the pages, use the scanner built into your phone instead. This exists for the photos you already have: the ones taken at an angle, on a kitchen table, in a hurry, that you now need as documents.

Everything happens on your machine. It makes no network calls.

## Install

```bash
pip3 install opencv-python numpy pillow
git clone https://github.com/jgalea/document-scanner.git
cd document-scanner
chmod +x scan.py
```

## Usage

```bash
# Every JPG in the current folder, one page per photo, out to ./scanned.pdf
./scan.py

# A folder of open-book photos, split each one at the binding
./scan.py ~/Downloads/book/ -o ~/Downloads/book.pdf --split-spreads

# Mixed: photos 0 and 6 are single-page covers, the rest are spreads
./scan.py ~/Downloads/book/ -o ~/Downloads/book.pdf --split-spreads --cover 0 --cover 6
```

## What it does to each photo

1. Loads it with EXIF rotation applied.
2. Finds the paper as the largest four-corner contour, trying edge detection first and brightness thresholding second.
3. Applies a perspective transform to flatten it, which removes the desk, the shadow along the paper's edge and the skew in one step.
4. With `--split-spreads`, finds the darkest vertical band near the centre and cuts there. That band is the binding.
5. Lifts the contrast mildly, 50 to 0 and 225 to 255. Pencil stays readable, and it avoids the bleached look that scanner apps produce.
6. Pads to A4 portrait on white.
7. Combines the pages into one PDF.

## Limits worth knowing

A single-page cover shot without a blank facing page can fool the contour detector, which sometimes picks a larger quadrilateral that includes the table. Mark those photos with `--cover INDEX`.

Thick books won't fully flatten, because a perspective transform assumes the page is flat and a heavily curved spread isn't.

It works best on photos taken from directly above, with the paper on a dark background, in reasonable light.

## License

MIT. See [LICENSE](LICENSE).

Author: Jean Galea.
