import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load the LLaMA model and tokenizer (you can download it from Hugging Face's model hub)
@st.cache_resource
def load_llama_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example: LLaMA 2 (hosted on Hugging Face)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_llama_model()

# Function to generate a response using LLaMA
def generate_llama_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Title of the app
st.title("PDF Invoice Querying App")

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
        # Combine extracted PDF text and the user query
        prompt = f"Here's the invoice text: '{pdf_text}'. Now, answer this query: {user_query}"

        # Generate response using LLaMA
        llama_response = generate_llama_response(prompt)

        # Display LLaMA's response
        st.subheader("Answer to your query")
        st.text(llama_response)
