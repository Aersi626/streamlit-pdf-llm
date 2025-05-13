import pandas as pd
import json

def format_table_as_markdown_json_hybrid(df: pd.DataFrame, page_num: int = None, table_num: int = None, table_name: str = "") -> str:
    """
    Combines markdown table, JSON data, and flattened key-value text for embedding.
    """
    markdown_str = df.to_markdown(index=False)

    flat_rows = []
    for i, row in df.iterrows():
        row_kv = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        flat_rows.append(f"Row {i + 1} - {row_kv}")
    flat_text = "\n".join(flat_rows)

    output = f"## {table_name or f'Table {table_num} on Page {page_num}'}\n\n"
    output += f"### Markdown View\n{markdown_str}\n\n"
    output += f"### Flattened View (for embedding)\n{flat_text}\n"

    return output
