
from pathlib import Path
import ast
import csv
import re

from openpyxl import load_workbook

# ============================================================
# E0.2b — PALETTE COMPLETENESS AUDIT
#
# Checks whether the colors omitted by E0.2 are present
# elsewhere in the official color-mapping workbook.
#
# Does not modify raw data or existing E0.2 results.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = (
    ROOT / "data/raw/PhenoLeaf-TS_DS"
    / "leaves_unique_colours.xlsx"
)

UNKNOWN_CSV = (
    ROOT / "experiments/E0/tables"
    / "E0_2_unknown_colors.csv"
)

OUTPUT = (
    ROOT / "experiments/E0/tables"
    / "E0_2b_palette_completeness.csv"
)

REPORT = (
    ROOT / "experiments/E0/reports"
    / "E0_2b_palette_report.md"
)

RGB_PATTERN = re.compile(
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?"
)


def parse_rgb(value):
    if value is None:
        return None

    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(v) for v in value)

    match = RGB_PATTERN.fullmatch(str(value).strip())

    if match:
        rgb = tuple(map(int, match.groups()))
        if all(0 <= v <= 255 for v in rgb):
            return rgb

    return None


def read_workbook():
    workbook = load_workbook(
        WORKBOOK,
        read_only=True,
        data_only=True,
    )

    print("Workbook sheets:", workbook.sheetnames)

    entries = []

    for sheet in workbook.worksheets:
        print(
            f"Sheet '{sheet.title}': "
            f"{sheet.max_row} rows"
        )

        for index, values in enumerate(
            sheet.iter_rows(values_only=True),
            start=1,
        ):
            for column, value in enumerate(values, start=1):
                rgb = parse_rgb(value)

                if rgb is not None:
                    leaf_number = (
                        values[2]
                        if len(values) > 2
                        else None
                    )

                    entries.append({
                        "sheet": sheet.title,
                        "row": index,
                        "column": column,
                        "rgb": rgb,
                        "leaf_number": leaf_number,
                        "row_contents": repr(values),
                    })

    workbook.close()
    return entries


def read_unknown_colors():
    results = []

    with UNKNOWN_CSV.open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            rgb = ast.literal_eval(row["rgb"])
            pixels = int(row["total_pixels"])

            if rgb == (0, 0, 0):
                continue

            results.append((rgb, pixels))

    return results


def main():
    entries = read_workbook()
    unknown = read_unknown_colors()

    by_rgb = {}

    for entry in entries:
        by_rgb.setdefault(
            entry["rgb"], []
        ).append(entry)

    print("\n" + "=" * 65)
    print("E0.2b — UNKNOWN COLOR INVESTIGATION")
    print("=" * 65)

    rows = []
    unresolved = []

    for rgb, pixels in unknown:
        matches = by_rgb.get(rgb, [])

        if not matches:
            status = "NOT_IN_WORKBOOK"
            unresolved.append(rgb)
        elif any(
            match["leaf_number"] is not None
            for match in matches
        ):
            status = "WORKBOOK_ENTRY_WITH_LEAF_NUMBER"
        else:
            status = "WORKBOOK_ENTRY_WITHOUT_LEAF_NUMBER"

        row = {
            "rgb": str(rgb),
            "pixels": pixels,
            "status": status,
            "workbook_locations": "; ".join(
                f"{m['sheet']} row {m['row']}"
                for m in matches
            ),
            "leaf_numbers": "; ".join(
                str(m["leaf_number"])
                for m in matches
                if m["leaf_number"] is not None
            ),
        }

        rows.append(row)

        print(
            f"{str(rgb):20s} "
            f"pixels={pixels:8d} "
            f"status={status}"
        )

        for match in matches:
            print(
                "   Workbook row:",
                match["row_contents"]
            )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rgb",
                "pixels",
                "status",
                "workbook_locations",
                "leaf_numbers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# E0.2b — Palette Completeness Audit",
        "",
        f"- Palette RGB entries found: {len(entries)}",
        f"- Unknown nonblack colors: {len(unknown)}",
        f"- Colors not found in workbook: {len(unresolved)}",
        "",
        "## Findings",
        "",
        "| RGB | Pixels | Status |",
        "|---|---:|---|",
    ]

    for row in rows:
        report.append(
            f"| {row['rgb']} | "
            f"{row['pixels']} | "
            f"{row['status']} |"
        )

    report.extend([
        "",
        "## Interpretation",
        "",
        "A matching workbook color is not by itself proof "
        "of a valid leaf identity.",
        "",
        "If unexplained colors remain, inspect their masks "
        "before excluding or assigning them.",
        "",
        "## Next",
        "",
        "Validate palette mapping and inspect example "
        "masks before freezing forecasting targets.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print("Workbook RGB entries:", len(entries))
    print("Unknown nonblack colors:", len(unknown))
    print("Not found in workbook:", len(unresolved))

    print("\nSaved:")
    print(OUTPUT)
    print(REPORT)


if __name__ == "__main__":
    main()
