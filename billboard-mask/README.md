# billboard-mask

Finds billboards in a photo and covers them:

- **Target billboard** (the biggest one, which is usually the closest): white fill with an orange border
- **All other billboards**: gray fill
- **Provider plaques / pole panels**: covered with a plain white box (no border), or with `--plaque blend` made to disappear: plaques on a plain wall get the wall colour, tall pole panels get the column's own colour, and plaques on a busy facade are filled in from their surroundings

```bash
pip install -r requirements.txt

python billboard_mask.py photo.jpg                      # writes photo_masked.jpg
python billboard_mask.py photos/ -o out/                # whole folder
python billboard_mask.py photo.jpg --debug              # also writes photo_debug.jpg showing what was detected
python billboard_mask.py photos/ -o out/ --mode haiku   # use Claude Haiku instead (needs ANTHROPIC_API_KEY)
```

## Modes

| | `--mode cv` (default) | `--mode haiku` |
|---|---|---|
| AI | none, plain OpenCV | Claude Haiku 5.5, thinking off, low effort |
| Cost | free, offline | ~$0.0002 per image (~5,000 images per $1) |
| Works well on | clear shots where the board has a visible rectangular frame against sky or a plain background | almost anything: cluttered streets, partly blocked boards, odd angles |
| Fails on | cluttered scenes (windows, trucks and shop signs look like rectangles too), boards with no contrasting frame, heavy occlusion | very small or distant boards; boxes can be a few % loose |

> **Tested on real roadside photos (Cyprus, Maltezos Media boards): CV mode failed on all three.** It boxed
> windows, a palm tree and a shop sign and missed the actual billboards, because the frames merge with
> poles and buildings in the edge map and dark ads blend into dark frames. Use `--mode haiku` for real photos;
> CV mode is only usable on very clean shots.

**How CV mode works:** edge detection → find 4-cornered, rectangle-like, landscape-shaped outlines → drop shapes nested inside bigger ones → keep the largest few. For plaques it searches a strip just above and below each board for a small, wide, solid sign, with the thin mounting pole filtered out.

## Options

- `--max N`: CV mode, the most billboards to keep per image (default 4)
- `--plaque white|blend`: how to hide the provider panel (default `white`)
- `--pad 0.03`: grow each cover by 3% so the board's frame edge is hidden too
- `--json`: also save the detected corners next to each output image
- `--boxes boxes.json`: skip detection and use boxes you supply (same shape Haiku returns), keyed by filename:
  `{"photo.jpg": {"billboards": [{"box": {"x1": 42, "y1": 7, "x2": 266, "y2": 174}, "plaque": {"x1": 132, "y1": 174, "x2": 186, "y2": 312}}]}}`
  For a board shot at an angle, give `"box"` as 4 corner points instead: `[[393, 139], [562, 161], [562, 351], [391, 350]]`
