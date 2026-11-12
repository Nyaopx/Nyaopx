#!/usr/bin/env python3
"""
Find billboards in a photo and cover them up.

  * Target billboard (the biggest one = usually the closest): white fill, orange border
  * Every other billboard: gray fill
  * Provider plaques (small sign stuck above/below a billboard): covered too,
    in the same style as the billboard they belong to

Two detectors:
  --mode cv     (default) pure OpenCV, no AI, free, offline. Works on clear shots
                where the billboard has a visible rectangular frame. Will miss some
                and occasionally grab other rectangles (windows, trucks, signs).
  --mode haiku  Claude Haiku 5.5, thinking off, low effort. Far more reliable.
                Roughly $0.0002 per image (~2k input tokens at $0.10/M).
                Needs ANTHROPIC_API_KEY.

Usage:
  python billboard_mask.py photo.jpg                     # -> photo_masked.jpg
  python billboard_mask.py photos/ -o out/ --mode haiku  # whole folder
  python billboard_mask.py photo.jpg --debug             # also save candidate overlay
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WHITE = (255, 255, 255)
ORANGE = (0, 140, 255)  # BGR
GRAY = (128, 128, 128)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Geometry helpers. A "quad" is a (4, 2) int array of corner points.
# --------------------------------------------------------------------------- #
def box_to_quad(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def quad_area(q):
    return cv2.contourArea(q.astype(np.float32))


def bbox(q):
    x, y, w, h = cv2.boundingRect(q.astype(np.int32))
    return x, y, x + w, y + h


def overlap_of_smaller(a, b):
    """Intersection area / area of the smaller quad (using bounding boxes)."""
    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    small = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return iw * ih / small if small else 0.0


# --------------------------------------------------------------------------- #
# Detector 1: classic computer vision (no AI)
# --------------------------------------------------------------------------- #
def find_quads(img, min_area_frac):
    """Return convex, rectangle-like 4-corner shapes found from edges."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)

    # Canny at two sensitivities (auto thresholds from median brightness):
    # normal catches crisp frames, sensitive catches low-contrast board edges.
    med = float(np.median(gray))
    contours = []
    for lo, hi in ((0.66, 1.33), (0.25, 0.6)):
        edges = cv2.Canny(gray, int(lo * med), int(min(255, hi * med)))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours += cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * w * h or area > 0.9 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(approx)
        if rw == 0 or rh == 0:
            continue
        rectangularity = cv2.contourArea(approx) / (rw * rh)
        aspect = max(rw, rh) / min(rw, rh)
        if rectangularity < 0.80 or aspect > 8:
            continue
        quads.append(approx.reshape(4, 2))
    return quads


def dedupe(quads, thresh=0.85):
    """Drop near-duplicates and anything sitting inside a bigger quad
    (inner panels, text boxes, the inside edge of a frame)."""
    quads = sorted(quads, key=quad_area, reverse=True)
    kept = []
    for q in quads:
        if all(overlap_of_smaller(q, k) < thresh for k in kept):
            kept.append(q)
    return kept


def looks_like_billboard(q):
    x1, y1, x2, y2 = bbox(q)
    w, h = x2 - x1, y2 - y1
    return 1.2 <= w / max(h, 1) <= 6.0  # billboards are landscape; tweak for portrait digital boards


def find_plaque(img, board):
    """Look in a band just below and just above the billboard for a small, wide,
    solid-looking sign (the provider plaque). The pole it's mounted on is thin
    and vertical, so a horizontal morphological opening removes it."""
    H, W = img.shape[:2]
    bx1, by1, bx2, by2 = bbox(board)
    bw, bh = bx2 - bx1, by2 - by1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    for side in ("below", "above"):
        if side == "below":
            y0, y1 = by2 + 2, min(H, by2 + int(0.45 * bh))
        else:
            y0, y1 = max(0, by1 - int(0.45 * bh)), by1 - 2
        x0, x1 = max(0, bx1), min(W, bx2)
        if y1 - y0 < 8 or x1 - x0 < 8:
            continue
        roi = gray[y0:y1, x0:x1]
        med = float(np.median(roi))
        edges = cv2.Canny(roi, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)) + 20)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(edges)
        cv2.drawContours(solid, cnts, -1, 255, cv2.FILLED)  # fill enclosed shapes
        kw, kh = max(9, int(0.06 * bw)), max(5, int(0.05 * bh))
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 1.5 * h or w > 0.7 * bw or w * h < 0.004 * bw * bh:
                continue
            if cv2.contourArea(c) < 0.75 * w * h:  # must be a filled rectangle
                continue
            gap = y if side == "below" else (y1 - y0) - (y + h)
            if gap > 0.2 * bh:
                continue
            if best is None or gap < best[0]:
                best = (gap, box_to_quad(x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return best[1] if best else None


def detect_cv(img, max_boards=4, debug_path=None):
    h, w = img.shape[:2]
    scale = 1280 / max(h, w) if max(h, w) > 1280 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img

    all_quads = find_quads(small, min_area_frac=0.003)
    big = [q for q in all_quads if quad_area(q) >= 0.003 * small.shape[0] * small.shape[1]]
    boards = [q for q in dedupe(big) if looks_like_billboard(q)][:max_boards]
    results = [{"board": b, "plaque": find_plaque(small, b)} for b in boards]

    if debug_path:
        dbg = small.copy()
        cv2.drawContours(dbg, [q.reshape(-1, 1, 2) for q in all_quads], -1, (0, 255, 0), 1)
        for r in results:
            cv2.polylines(dbg, [r["board"].reshape(-1, 1, 2)], True, (0, 0, 255), 3)
            if r["plaque"] is not None:
                cv2.polylines(dbg, [r["plaque"].reshape(-1, 1, 2)], True, (255, 0, 255), 3)
        cv2.imwrite(str(debug_path), dbg)

    # Scale back to original resolution
    for r in results:
        r["board"] = (r["board"] / scale).astype(np.int32)
        if r["plaque"] is not None:
            r["plaque"] = (r["plaque"] / scale).astype(np.int32)
    return results


# --------------------------------------------------------------------------- #
# Detector 2: Claude Haiku 5.5 (cheap, much more robust)
# --------------------------------------------------------------------------- #
HAIKU_PROMPT = (
    "Find every billboard in this image (roadside/rooftop advertising boards, "
    "static or digital). For each, give its pixel bounding box in THIS image "
    "({w}x{h}), covering the whole board face including its frame. If the "
    "billboard has a provider/owner name panel (often a tall vertical panel on "
    "the support pole under the board with the media company name, or a small "
    "nameplate above/below the board), give a box covering that panel too, else "
    "null. Do not include street signs, shop signs on building facades, or "
    "vehicles. Return an empty list if there are none."
)
BOX = {
    "type": "object",
    "properties": {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")},
    "required": ["x1", "y1", "x2", "y2"],
    "additionalProperties": False,
}
HAIKU_SCHEMA = {
    "type": "object",
    "properties": {
        "billboards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"box": BOX, "plaque": {"anyOf": [BOX, {"type": "null"}]}},
                "required": ["box", "plaque"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["billboards"],
    "additionalProperties": False,
}


def detect_haiku(img, client):
    h, w = img.shape[:2]
    # Send at most 1568px on the long side: that's what the model sees anyway,
    # and the coordinates it returns are in this resized space.
    scale = min(1.0, 1568 / max(h, w))
    send = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    sh, sw = send.shape[:2]
    ok, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.standard_b64encode(buf.tobytes()).decode()

    resp = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        thinking={"type": "disabled"},
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": HAIKU_SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": HAIKU_PROMPT.format(w=sw, h=sh)},
            ],
        }],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)

    u = resp.usage
    cost = u.input_tokens * 0.10 / 1e6 + u.output_tokens * 0.50 / 1e6
    print(f"  haiku: {u.input_tokens} in / {u.output_tokens} out tokens, ~${cost:.5f}")
    return results_from_boxes(data, scale)


def results_from_boxes(data, scale=1.0):
    """{"billboards": [{"box": {x1,y1,x2,y2}, "plaque": {...} | null}, ...]} -> results"""
    def to_quad(b):
        if isinstance(b, list):  # 4 corner points [[x, y], ...] for angled boards
            return (np.array(b, dtype=np.float32) / scale).astype(np.int32)
        x1, x2 = sorted((b["x1"], b["x2"]))
        y1, y2 = sorted((b["y1"], b["y2"]))
        return (box_to_quad(x1, y1, x2, y2) / scale).astype(np.int32)

    return [
        {"board": to_quad(bb["box"]), "plaque": to_quad(bb["plaque"]) if bb.get("plaque") else None}
        for bb in data["billboards"]
    ]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def pad_quad(q, img_shape, frac=0.03):
    """Grow a quad a little around its centre so the cover hides the frame edge."""
    c = q.mean(axis=0)
    out = c + (q - c) * (1 + frac)
    h, w = img_shape[:2]
    out[:, 0] = out[:, 0].clip(0, w - 1)
    out[:, 1] = out[:, 1].clip(0, h - 1)
    return out.astype(np.int32)


def blend_colour(img, q, boards_mask):
    """Colour that makes a covered plaque disappear, or None = inpaint it.
    Plaque on a plain wall -> use the wall colour around it.
    Tall panel on a pole   -> use the panel's own colour, so the column looks blank.
    Wide plaque on a busy facade -> None: fill it in from its surroundings."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox(q)
    m = max(4, (y2 - y1) // 3)
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    ring = np.ones((Y2 - Y1, X2 - X1), bool)
    ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    ring &= ~boards_mask[Y1:Y2, X1:X2]  # ignore the billboard the plaque hangs off
    around = img[Y1:Y2, X1:X2][ring].astype(np.float32)
    # median absolute deviation: plain wall stays low even with a window edge in view
    if len(around) > 20 and np.median(np.abs(around - np.median(around, axis=0)), axis=0).mean() < 10:
        return tuple(int(c) for c in np.median(around, axis=0))
    if y2 - y1 > x2 - x1:
        # text/logo is a minority of the panel, so the median is the panel colour
        return tuple(int(c) for c in np.median(img[y1:y2, x1:x2].reshape(-1, 3), axis=0))
    return None


def cover(img, results, pad=0.03, plaque_style="blend"):
    """plaque_style: "white" = plain white fill, no border (hides the provider name)
                     "blend" = fill with the panel's own dominant colour, so the
                               pole/column just looks blank"""
    out = img.copy()
    if not results:
        return out

    boards_mask = np.zeros(img.shape[:2], np.uint8)
    for r in results:
        cv2.fillPoly(boards_mask, [pad_quad(r["board"], img.shape, pad).reshape(-1, 1, 2)], 1)
    boards_mask = boards_mask.astype(bool)

    # Plaques first, so the billboard covers sit cleanly on top where they touch.
    for r in results:
        if r["plaque"] is None:
            continue
        q = pad_quad(r["plaque"], img.shape, pad)
        colour = blend_colour(img, q, boards_mask) if plaque_style == "blend" else WHITE
        if colour is None:
            hole = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(hole, [q.reshape(-1, 1, 2)], 255)
            out = cv2.inpaint(out, cv2.dilate(hole, np.ones((5, 5), np.uint8)), 9, cv2.INPAINT_NS)
        else:
            cv2.fillPoly(out, [q.reshape(-1, 1, 2)], colour)

    # Target = biggest billboard (biggest on screen is almost always the closest)
    target_i = max(range(len(results)), key=lambda i: quad_area(results[i]["board"]))
    order = [i for i in range(len(results)) if i != target_i] + [target_i]  # target drawn last, on top
    for i in order:
        q = pad_quad(results[i]["board"], img.shape, pad)
        pts = q.reshape(-1, 1, 2)
        if i == target_i:
            cv2.fillPoly(out, [pts], WHITE)
            thick = max(3, int(0.03 * min(bbox(q)[2] - bbox(q)[0], bbox(q)[3] - bbox(q)[1])))
            cv2.polylines(out, [pts], True, ORANGE, thick, cv2.LINE_AA)
        else:
            cv2.fillPoly(out, [pts], GRAY)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder of images")
    ap.add_argument("-o", "--output", help="output file or folder (default: <name>_masked.<ext>)")
    ap.add_argument("--mode", choices=["cv", "haiku"], default="cv")
    ap.add_argument("--max", type=int, default=4, help="cv mode: max billboards per image")
    ap.add_argument("--pad", type=float, default=0.03, help="grow covers by this fraction")
    ap.add_argument("--plaque", choices=["blend", "white"], default="blend",
                    help="provider panel cover: blend it away (default), or a plain white box")
    ap.add_argument("--debug", action="store_true", help="cv mode: save *_debug.jpg with candidates")
    ap.add_argument("--json", action="store_true", help="also write detected corners as *.json")
    ap.add_argument("--boxes", help="skip detection: JSON file of boxes per image filename, "
                    'e.g. {"photo.jpg": {"billboards": [{"box": {"x1":..,"y1":..,"x2":..,"y2":..}, "plaque": null}]}}')
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out_dir = Path(args.output) if args.output and (src.is_dir() or Path(args.output).suffix == "") else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    given = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    if args.mode == "haiku" and not given:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip {f}: can't read")
            continue
        if out_dir:
            dst = out_dir / f"{f.stem}_masked{f.suffix}"
        elif args.output:
            dst = Path(args.output)
        else:
            dst = f.with_name(f"{f.stem}_masked{f.suffix}")

        print(f"{f.name}:")
        try:
            if given is not None:
                results = results_from_boxes(given.get(f.name, {"billboards": []}))
            elif args.mode == "cv":
                dbg = dst.with_name(f"{f.stem}_debug.jpg") if args.debug else None
                results = detect_cv(img, args.max, dbg)
            else:
                results = detect_haiku(img, client)
        except Exception as e:  # keep going on a batch
            print(f"  error: {e}")
            continue

        cv2.imwrite(str(dst), cover(img, results, args.pad, args.plaque))
        plaques = sum(r["plaque"] is not None for r in results)
        print(f"  {len(results)} billboard(s), {plaques} plaque(s) -> {dst}")
        if args.json:
            dst.with_suffix(".json").write_text(json.dumps(
                [{"board": r["board"].tolist(), "plaque": None if r["plaque"] is None else r["plaque"].tolist()}
                 for r in results], indent=1))


if __name__ == "__main__":
    sys.exit(main())
