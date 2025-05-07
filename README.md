# Streamlit PDF Reader with LLM Integration

This app extracts text from PDFs and processes it using OpenAI's language models.

## Features
- Extract text from uploaded PDF files.
- Summarize or analyze the extracted text using OpenAI.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Aersi626/streamlit-pdf-llm.git
   cd streamlit-pdf-llm

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

4. Ollama server
    ```bash
    ollama pull llama3
    ollama serve

5. Run the app:
    ```bash
    streamlit run app/ui.py