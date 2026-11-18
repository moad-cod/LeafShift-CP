
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import csv
import random

# ============================================================
# E0.12 — MANUAL LEAF-IDENTITY VALIDATION
#
# 3 detector-positive cases + 3 detector-negative controls
# Read-only access to the original dataset.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"

RAW = DATA / "01_raw_dataset"
LABELS = DATA / "02_leaf_labels"

TABLES = ROOT / "experiments/E0/tables"
OUT = ROOT / "experiments/E0/figures/E0_12_manual_review"

OUT.mkdir(parents=True, exist_ok=True)

TRANSITIONS = TABLES / "E0_11_sampled_transitions.csv"
MATCHES = TABLES / "E0_11_candidate_matches.csv"

RANDOM_SEED = 20261108
PANEL_WIDTH = 420
PANEL_HEIGHT = 420


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(
        int(value[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def highlight_color(image, color_hex):
    rgb = hex_to_rgb(color_hex)

    img = image.convert("RGB")
    pixels = img.load()

    mask = Image.new("L", img.size, 0)
    selected = mask.load()

    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y] == rgb:
                selected[x, y] = 255

    return mask


def overlay(raw, mask_image, color_hex):
    mask = highlight_color(mask_image, color_hex)

    result = raw.convert("RGB")
    magenta = Image.new(
        "RGB", result.size, (255, 0, 255)
    )
    result.paste(magenta, (0, 0), mask)

    bbox = mask.getbbox()

    if bbox:
        draw = ImageDraw.Draw(result)
        draw.rectangle(
            bbox,
            outline=(255, 255, 0),
            width=5,
        )

    return result


def panel(image, label):
    canvas = Image.new(
        "RGB",
        (PANEL_WIDTH, PANEL_HEIGHT + 30),
        "white",
    )

    image = image.copy()
    image.thumbnail(
        (PANEL_WIDTH, PANEL_HEIGHT)
    )

    x = (PANEL_WIDTH - image.width) // 2
    y = 30 + (PANEL_HEIGHT - image.height) // 2

    canvas.paste(image, (x, y))
    ImageDraw.Draw(canvas).text(
        (10, 8), label, fill="black"
    )
    return canvas


def build_review(item, match, output):
    genotype, replicate = item["plant_id"].split("/")

    before_name = item["previous_filename"]
    after_name = item["current_filename"]

    with Image.open(
        RAW / genotype / replicate / before_name
    ) as im:
        before_raw = im.convert("RGB")

    with Image.open(
        RAW / genotype / replicate / after_name
    ) as im:
        after_raw = im.convert("RGB")

    with Image.open(
        LABELS / genotype / replicate / before_name
    ) as im:
        before_mask = im.convert("RGB")

    with Image.open(
        LABELS / genotype / replicate / after_name
    ) as im:
        after_mask = im.convert("RGB")

    panels = [
        panel(before_raw, "BEFORE: raw"),
        panel(before_mask, "BEFORE: labels"),
        panel(after_raw, "AFTER: raw"),
        panel(after_mask, "AFTER: labels"),
    ]

    if match:
        panels[1] = panel(
            overlay(
                before_raw,
                before_mask,
                match["old_color"],
            ),
            "BEFORE: suspected old identity",
        )
        panels[3] = panel(
            overlay(
                after_raw,
                after_mask,
                match["new_color"],
            ),
            "AFTER: suspected new identity",
        )

    sheet = Image.new(
        "RGB",
        (2 * PANEL_WIDTH, 2 * (PANEL_HEIGHT + 30) + 40),
        "#dddddd",
    )

    for i, image in enumerate(panels):
        x = (i % 2) * PANEL_WIDTH
        y = 40 + (i // 2) * (PANEL_HEIGHT + 30)
        sheet.paste(image, (x, y))

    title = (
        f"{item['plant_id']} | "
        f"Detector: {'POSITIVE' if match else 'NEGATIVE'}"
    )

    ImageDraw.Draw(sheet).text(
        (10, 12), title, fill="black"
    )

    sheet.save(output)


def main():
    transitions = read_csv(TRANSITIONS)
    matches = read_csv(MATCHES)

    key_fields = (
        "plant_id",
        "previous_filename",
        "current_filename",
    )

    def key(row):
        return tuple(row[field] for field in key_fields)

    by_transition = defaultdict(list)

    for match in matches:
        by_transition[key(match)].append(match)

    positive = [
        row for row in transitions
        if int(row["has_strong_match"]) == 1
    ]

    negative = [
        row for row in transitions
        if int(row["has_strong_match"]) == 0
    ]

    rng = random.Random(RANDOM_SEED)

    # Use all three detected positives; choose three
    # negative examples from different genotypes.
    selected = list(positive[:3])
    rng.shuffle(negative)

    used_genotypes = set()

    for row in negative:
        genotype = row["genotype"]

        if genotype in used_genotypes:
            continue

        selected.append(row)
        used_genotypes.add(genotype)

        if len(selected) >= 6:
            break

    index = []

    for i, item in enumerate(selected, start=1):
        match_list = by_transition.get(
            key(item), []
        )
        match = match_list[0] if match_list else None

        filename = f"review_{i:02d}.png"

        build_review(
            item,
            match,
            OUT / filename,
        )

        index.append({
            "review_image": filename,
            "genotype": item["genotype"],
            "plant_id": item["plant_id"],
            "previous_filename": item["previous_filename"],
            "current_filename": item["current_filename"],
            "detector_positive": int(bool(match)),
            "old_color": match["old_color"] if match else "",
            "new_color": match["new_color"] if match else "",
            "manual_label": "PENDING",
            "manual_notes": "",
        })

        print(
            f"{filename}: {item['plant_id']} "
            f"detector_positive={bool(match)}"
        )

    save_csv(
        OUT / "review_index.csv",
        index,
        list(index[0].keys()),
    )

    print("\nE0.12 complete.")
    print("Review images:", OUT)
    print("Review table:", OUT / "review_index.csv")


if __name__ == "__main__":
    main()
