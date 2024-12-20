import streamlit as st
import PyPDF2
import openai

# streamlit app title
st.title("PDF Reader with LLM Inegration")
st.write("Upload a PDF and let an LLM process its contents, extract the information from the PDF file and ask question about the file")

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
        "Keywords": "Extrace the main keywords from the following text."
    }

    if task == "Summarize":
        prompt = f"{task_templates[task][style]}\n\nText:\n{text}"
    elif task == "keywords":
        prompt = f"{task_templates[task]}\n\nText:\n{text}"
    else:
        pass
    
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

def ask_question_with_llm(text, question):
    messages = [
        {"role": "system", "content": "You are an expert assistant who answers questions based on the given context."},
        {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2000,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# Initialize session state for FPD procesing
if "pdf_processing_done" not in st.session_state:
    st.session_state.pdf_processing_done = False

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None  # To store the summary result
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # To store question-answer pairs

# file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# extrace text from PDF
if uploaded_file and not st.session_state.pdf_processing_done:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        # store the text in session state
        st.session_state.pdf_text = pdf_text

    # display extracted text
    st.subheader("Extracted Text (Preview)")
    st.write(pdf_text[:500]+ "...")

    # process with LLM
if st.session_state.pdf_text and st.button("Summarize PDF"):
    with st.spinner("Processing text with LLM..."):
        llm_output = process_text_with_llm(pdf_text)
        # Save summary result to session state
        st.session_state.summary_result = llm_output

# display LLM output
if st.session_state.summary_result:
    st.subheader("LLM Output")
    st.write(st.session_state.summary_result)

# ask question from the pdf
if st.session_state.pdf_text:
    st.subheader("Ask Questions About the PDF")
    user_question = st.text_input("Enter your question to this file:")
    
    if st.button("Ask Question") and user_question:
        with st.spinner("Thinking..."):
            answer = ask_question_with_llm(st.session_state.pdf_text, user_question)
        st.session_state.qa_history.append((user_question, answer))
        
# Display all questions and answers
if st.session_state.qa_history:
    st.subheader("Q&A History")
    for i, (question, answer) in enumerate(st.session_state.qa_history, 1):
        st.write(f"**Question{i}:** {question}")
        st.write(f"**Answer{i}:** {answer}")
        st.write("---")  # Add a separator for readability
        
st.write("Made with Heart using Streamlit and OpenAI")