import pdfplumber
import pandas as pd
from typing import List, Tuple
import re
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "../config/table_headers_config.json")
with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
    HEADER_CONFIG = json.load(f)

def extract_section_key(caption: str) -> str:
    match = re.match(r"(\d+(?:\.\d+){0,2})", caption)
    return match.group(1) if match else ""

def fix_multirow_header(rows: list[list[str]], expected_min_cols: int = 4, max_rows: int = 3) -> tuple[list[str], list[list[str]]]:
    """
    Combine multiple header rows into a single row if the first row appears malformed.
    Returns: (fixed_header, remaining_rows)
    """
    header_candidate = rows[:max_rows]
    num_cols = max(len(row) for row in header_candidate)

    # Combine headers
    combined = ['' for _ in range(num_cols)]
    for row in header_candidate:
        for i in range(len(row)):
            val = (row[i] or "").strip()
            if val:
                combined[i] += (" " + val).strip() if combined[i] else val

    cleaned_header = [h.strip() for h in combined if h.strip()]

    # Sanity check: return combined header only if it's clearly more structured
    if len(cleaned_header) >= expected_min_cols:
        return cleaned_header, rows[max_rows:]
    else:
        return rows[0], rows[1:]

def log_raw_table(table: list[list[str]], page_num: int, table_idx: int, caption: str = "", log_dir="logs/raw_tables"):
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"page_{page_num}_table_{table_idx}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Page: {page_num}, Table: {table_idx}, Caption: {caption}\n\n")
            for row in table:
                f.write(" | ".join(str(cell) if cell is not None else "[None]" for cell in row) + "\n")

def extract_tables_from_pdf(pdf_path: str) -> List[Tuple[pd.DataFrame, int, int, str]]:
    """
    Extracts tables and assigns them to the most recent section heading.
    Returns list of (DataFrame, start_page_num, table_index, section_heading).
    """
    tables = []
    current_caption = None
    current_header = None
    current_rows = []
    current_col_count = None
    start_page = None
    table_index = 0

    # Match patterns like "8.3 Status Variables", "10 Equipment Constants"
    section_heading_pattern = re.compile(
    r"^(\d+(?:\.\d+){0,2})\s+([A-Z][\w\-]*(?:\s+[a-zA-Z][\w\-]*){0,5})",
    re.MULTILINE)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            section_matches = section_heading_pattern.findall(text)

            section_key = None
            for sec, _ in section_matches:
                if sec in HEADER_CONFIG:
                    section_key = sec
                    break  # use the first matching section key found
            section_caption = section_key if section_key else current_caption

            page_tables = page.extract_tables()
            for table in page_tables:
                print(f"🔍 Processing table {table_index} on page {page_num} under caption: {section_caption}")
                log_raw_table(table, page_num, table_index, section_caption or "Unknown")

                if not table or len(table) < 2 or not any(cell for row in table for cell in row):
                    print(f"⚠️ Skipping empty or malformed table {table_index} on page {page_num}")
                    continue

                print(f"🔎 Raw first row of table {table_index} on page {page_num}: {table[0]}")

                section_key = extract_section_key(section_caption or "")
                known_header = HEADER_CONFIG.get(section_key)
                print(f"Section Key: {section_key}, Known Header: {known_header}")

                if known_header:
                    maybe_header = known_header
                    rest_rows = table  # use entire table as body
                    print(f"⚠️ Using known header for section {section_key}: {maybe_header}")
                else:
                    if len([c for c in table[0] if c and c.strip()]) < 3:
                        maybe_header, rest_rows = fix_multirow_header(table)
                    else:
                        maybe_header = table[0]
                        rest_rows = table[1:]

                if not maybe_header or not any(c.strip() for c in maybe_header if c):
                    print(f"❌ Invalid or empty header detected at page {page_num}, table {table_index}")
                    continue
                else:
                    if sum(1 for c in table[0] if c and str(c).strip()) < 3:
                        maybe_header, rest_rows = fix_multirow_header(table)
                    else:
                        maybe_header = table[0]
                        rest_rows = table[1:]

                if section_caption != current_caption:
                    for i, row in enumerate(current_rows):
                        if len(row) != len(current_header):
                            print(f"⚠️ Skipping malformed row at table {table_index}, page {start_page}: {row} {len(row)} {current_header} {len(current_header)}")
                    # New section detected — finalize previous table
                    cleaned_rows = [row for row in current_rows if len(row) == len(current_header)]
                    if cleaned_rows:
                        df = pd.DataFrame(cleaned_rows, columns=current_header)
                        tables.append((df, start_page, table_index, current_caption))
                        table_index += 1

                    # Start a new table group
                    current_caption = section_caption
                    current_header = maybe_header
                    current_rows = rest_rows
                    current_col_count = len(maybe_header)
                    start_page = page_num
                else:
                    # Same section: continue previous table
                    if maybe_header == current_header or len(maybe_header) == current_col_count:
                        current_rows.extend(rest_rows)
                    else:
                        current_rows.extend(table)

    if current_rows:
        cleaned_rows = [row for row in current_rows if len(row) == len(current_header)]
        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows, columns=current_header)
            tables.append((df, start_page, table_index, current_caption))
        else:
            print(f"⚠️ No clean rows found for table {table_index} on page {start_page}")

    return tables

def extract_text_without_tables(page):
    # Start with full page
    text_parts = []
    full_bbox = page.bbox  # (x0, top, x1, bottom)

    # Get table boxes
    tables = page.find_tables()
    table_bboxes = [table.bbox for table in tables]

    # Invert the boxes: extract text from non-table regions
    y_coords = sorted(set([full_bbox[1]] + [b[1] for b in table_bboxes] + [b[3] for b in table_bboxes] + [full_bbox[3]]))

    for i in range(len(y_coords) - 1):
        y0, y1 = y_coords[i], y_coords[i + 1]
        slice_box = (full_bbox[0], y0, full_bbox[2], y1)

        # Only include if not overlapping any table
        if not any(overlap_y(slice_box, b) for b in table_bboxes):
            cropped = page.within_bbox(slice_box)
            text = cropped.extract_text()
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)

def overlap_y(a, b):
    return not (a[3] <= b[1] or a[1] >= b[3])