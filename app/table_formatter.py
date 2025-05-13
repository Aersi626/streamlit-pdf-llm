import pandas as pd
import json

def format_table_as_markdown_json_hybrid(df: pd.DataFrame, page_num: int = None, table_num: int = None) -> str:
    """
    Converts a pandas DataFrame to a hybrid Markdown + JSON string for better LLM comprehension.
    """
    markdown = df.to_markdown(index=False)
    json_data = df.to_dict(orient="records")
    json_str = json.dumps(json_data, indent=2)

    header = f"### Table {table_num} on Page {page_num}\n" if page_num and table_num else ""

    return (
        f"{header}Below is the same table in two formats: Markdown (for readability) and JSON (for structure).\n\n"
        f"#### Markdown Table:\n```\n{markdown}\n```\n\n"
        f"#### JSON Table:\n```json\n{json_str}\n```\n"
    )
