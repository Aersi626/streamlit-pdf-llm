import pdfplumber
import pandas as pd
from typing import List, Tuple

def extract_tables_from_pdf(pdf_path: str) -> List[Tuple[pd.DataFrame, int, int]]:
    """
    Extract tables from a PDF as a list of (DataFrame, page_num, table_num) tuples.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List of tuples containing:
            - DataFrame of the table
            - Page number
            - Table number on that page
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            for table_num, table in enumerate(page_tables, start=1):
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append((df, page_num, table_num))
                except Exception as e:
                    print(f"⚠️ Failed to parse table on page {page_num}, table {table_num}: {e}")
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