import streamlit as st
import fitz
from google.generativeai import chat

# Fetch API key from Streamlit secrets
api_key = st.secrets["google_gemini"]["api_key"]

# Authenticate the Google Gemini API using the API key
chat.configure(api_key=api_key)

# Example setup of the ChatModel
chat_model = chat.ChatModel()

# Rest of your code...
st.title("PDF Invoice Querying App")

uploaded_file = st.file_uploader("Upload a PDF invoice", type=["pdf"])

if uploaded_file:
    # Extract and process the PDF content as before

    # User can ask a question related to the PDF
    user_query = st.text_input("Ask a query related to the invoice")

    if user_query:
        # Send extracted text and user query to the Gemini model for a response
        message = {
            "prompt": f"Here's the invoice text: '{pdf_text}'. Now, answer this query: {user_query}"
        }
        gemini_response = chat_model.generate_chat(message=message)

        # Display the Gemini API's response
        st.subheader("Answer to your query")
        st.text(gemini_response["response"])
