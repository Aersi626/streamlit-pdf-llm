import streamlit as st
import PyPDF2
import openai

# streamlit app title
st.title("PDF Reader with LLM Inegration")
st.write("Upload a PDF and let an LLM process its contents, extract the information from the PDF file")

# st.write("Secrets:", st.secrets)  # Debugging line
# secure OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# function to extrace taxt from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# function to process text with OpenAI LLM
def process_text_with_llm(text, task="Summarize", style="Bullet Points", temperature=0.7):
    task_templates = {
        "Summarize":{
            "Bullet Points": "Summarize the text into concise bullet points.",
            "Narrative": "Summarize the text in a narrative format.",
            "Abstract": "Provide an abstract-like summary of the text."
        },
        "Keywords": "Extrace the main keywords from the following text.",
        "Q&A": "Based on the following text, answer the question."
    }

    if task == "Summarize":
        prompt = f"{task_templates[task][style]}\n\nText:\n{text}"
    elif task == "keywords":
        prompt = f"{task_templates[task]}\n\nText:\n{text}"
    else:
        question = st.text_input("Enter your qeustion:")
        prompt = f"{task_templates[task]}\n\nQuestion:{question}\n\nText:\n{text}"
    
    messages=[
            {"role": "system", "content": "You are a highly skilled assistant specializing in text processing."},
            {"role": "user", "content": prompt}
        ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2000,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # extrace text from PDF
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    # display extracted text
    st.subheader("Extracted Text")
    st.write(pdf_text[:500])

    # process with LLM
    with st.spinner("Processing text with LLM..."):
        llm_output = process_text_with_llm(pdf_text)

    # display LLM output
    st.subheader("LLM Output")
    st.write(llm_output)

st.write("Made with Heart using Streamlit and OpenAI")