
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageChops
from openpyxl import load_workbook
import csv
import re

# ============================================================
# E0.8 — VISUAL LEAF ANOMALY AUDIT
#
# Original data is read-only.
# Selects extreme events from multiple plants.
#
# Output: visual comparisons and review index.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"
RAW = DATA / "01_raw_dataset"
LABELS = DATA / "02_leaf_labels"
PALETTE = DATA / "leaves_unique_colours.xlsx"

EVENTS = (
    ROOT / "experiments/E0/tables"
    / "E0_7_suspicious_leaf_events.csv"
)

OUT = (
    ROOT / "experiments/E0/figures"
    / "E0_8_visual_review"
)
OUT.mkdir(parents=True, exist_ok=True)

MAX_PER_TYPE = 4
PANEL_SIZE = (350, 350)

RGB_PATTERN = re.compile(
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?"
)


def parse_rgb(value):
    match = RGB_PATTERN.fullmatch(str(value).strip())
    if not match:
        return None
    rgb = tuple(map(int, match.groups()))
    return rgb if all(0 <= x <= 255 for x in rgb) else None


def load_palette():
    book = load_workbook(
        PALETTE, read_only=True, data_only=True
    )
    sheet = book["Leaf Colour Palette"]
    mapping = {}

    for row in sheet.iter_rows(
        min_row=2, values_only=True
    ):
        if len(row) < 3:
            continue

        rgb = parse_rgb(row[1])
        try:
            leaf_id = int(row[2])
        except (ValueError, TypeError):
            continue

        if rgb is not None:
            mapping[leaf_id] = rgb

    book.close()
    return mapping


def target_mask(mask_image, rgb):
    r, g, b = mask_image.split()

    def equal_channel(channel, value):
        return channel.point(
            lambda x: 255 if x == value else 0
        )

    result = ImageChops.multiply(
        equal_channel(r, rgb[0]),
        equal_channel(g, rgb[1])
    )
    result = ImageChops.multiply(
        result,
        equal_channel(b, rgb[2])
    )
    return result


def make_panel(image, title, width=350):
    image = image.copy()
    image.thumbnail(PANEL_SIZE)

    panel = Image.new(
        "RGB", (width, 390), "white"
    )
    x = (width - image.width) // 2
    y = 35 + (350 - image.height) // 2
    panel.paste(image, (x, y))

    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), title, fill="black")
    return panel


def render_frame(raw_file, mask_file, rgb):
    with Image.open(raw_file) as image:
        raw = image.convert("RGB")

    with Image.open(mask_file) as image:
        label = image.convert("RGB")

    if raw.size != label.size:
        raise ValueError(
            f"Image/mask dimensions differ: {mask_file}"
        )

    selection = target_mask(label, rgb)
    pixel_count = sum(selection.histogram()[255:256])

    overlay = raw.copy()
    highlight = Image.new("RGB", raw.size, (255, 0, 255))
    overlay.paste(highlight, (0, 0), selection)

    bbox = selection.getbbox()
    if bbox:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            bbox, outline=(255, 255, 0), width=5
        )

    return [
        make_panel(raw, "Raw image"),
        make_panel(label, "Full leaf-label mask"),
        make_panel(
            overlay,
            f"Target leaf highlighted ({pixel_count} px)"
        ),
    ]


def make_comparison(event, rgb, path):
    genotype = event["genotype"]
    replicate = event["plant_id"].split("/")[-1]

    filenames = [
        event["previous_filename"],
        event["current_filename"],
    ]

    canvas = Image.new(
        "RGB", (1050, 850), "#eeeeee"
    )
    draw = ImageDraw.Draw(canvas)

    for i, filename in enumerate(filenames):
        raw_file = RAW / genotype / replicate / filename
        mask_file = LABELS / genotype / replicate / filename

        panels = render_frame(raw_file, mask_file, rgb)

        y = 35 + i * 405
        for j, panel in enumerate(panels):
            canvas.paste(panel, (j * 350, y))

        draw.text(
            (10, y - 22),
            "BEFORE" if i == 0 else "AFTER",
            fill="black"
        )

    canvas.save(path)


def main():
    palette = load_palette()
    print("Mapped leaf colors:", len(palette))

    with EVENTS.open(
        newline="", encoding="utf-8"
    ) as file:
        events = list(csv.DictReader(file))

    selected = []
    used = set()

    for event_type in (
        "SHARP_DECREASE",
        "SHARP_INCREASE"
    ):
        count = 0

        for event in events:
            if event["event"] != event_type:
                continue

            plant_id = event["plant_id"]
            if plant_id in used:
                continue

            leaf_id = int(event["leaf_id"])
            if leaf_id not in palette:
                continue

            selected.append(event)
            used.add(plant_id)
            count += 1

            if count >= MAX_PER_TYPE:
                break

    # Include an explicit increase/decrease example
    # from Ler-1 if not already selected.
    if not any(
        x["plant_id"] == "Ler-1/rep_20"
        for x in selected
    ):
        for event in events:
            if event["plant_id"] == "Ler-1/rep_20":
                if int(event["leaf_id"]) in palette:
                    selected.append(event)
                    break

    index_rows = []

    for i, event in enumerate(selected, start=1):
        leaf_id = int(event["leaf_id"])
        rgb = palette[leaf_id]

        output = OUT / f"event_{i:02d}.png"

        make_comparison(event, rgb, output)

        index_rows.append({
            "image": output.name,
            "plant_id": event["plant_id"],
            "leaf_id": leaf_id,
            "event": event["event"],
            "old_pixels": event["previous_area_pixels"],
            "new_pixels": event["current_area_pixels"],
            "time_gap_hours": event["time_gap_hours"],
            "review_status": "PENDING_VISUAL_REVIEW",
        })

        print(
            f"{output.name}: "
            f"{event['plant_id']} "
            f"leaf={leaf_id}, "
            f"{event['previous_area_pixels']} -> "
            f"{event['current_area_pixels']} px"
        )

    columns = list(index_rows[0].keys())

    with (OUT / "review_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print("\nE0.8 complete.")
    print("Visualizations saved in:", OUT)


if __name__ == "__main__":
    main()