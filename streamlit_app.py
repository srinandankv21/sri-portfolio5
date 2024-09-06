import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load T5 model and tokenizer
@st.cache_resource
def load_t5_model():
    model_name = "t5-small"  # You can use other variants like 't5-base' or 't5-large'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_t5_model()

# Function to generate a response using T5
def generate_t5_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Title of the app
st.title("PDF Invoice Querying App using T5")

# Allow user to upload a PDF invoice
uploaded_file = st.file_uploader("Upload a PDF invoice", type=["pdf"])

if uploaded_file:
    # Extract text from the PDF file
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_document:
        pdf_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pdf_text += page.get_text("text")  # Extract text from each page

    # Display the extracted text
    st.subheader("Extracted Invoice Text")
    st.text(pdf_text)
    
    # Input box to ask a query
    user_query = st.text_input("Ask a query related to the invoice")

    if user_query:
        # Prepare the input prompt for T5
        prompt = f"Invoice text: {pdf_text}\nQuestion: {user_query}"

        # Generate response using T5
        t5_response = generate_t5_response(prompt)

        # Display T5's response
        st.subheader("Answer to your query")
        st.text(t5_response)
