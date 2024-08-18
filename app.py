import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Replace this
# from langchain.document_loaders import PyPDFLoader, TextLoader
# with following
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from transformers import AutoTokenizer, AutoModelForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import base64
import os
from gtts import gTTS
import io

# Model and tokenizer loading
checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
device = torch.device("cpu")
base_model.to(device)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto')

# Summarization pipeline
def summarize_text(text):
    summarizer = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=194,
        min_length=50
    )
    result = summarizer(text)
    return result[0]["summary_text"]

# File preprocessing
import mimetypes
def preprocess_file(file):
    mime_type, _ = mimetypes.guess_type(file)
    # if file.type == "application/pdf":
    if mime_type == "application/pdf":
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
    else:
        # For other types of text files
        loader = TextLoader(file)
        pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    final_texts = [text.page_content for text in texts]
    
    return ' '.join(final_texts)  # Join text chunks for summarization

# Sentiment Analysis (Dummy function, as sentiment analysis is different from summarization)
def sentiment_analysis(text):
    # This is a placeholder function for sentiment analysis
    # You would need a proper sentiment analysis model or API for real use
    return "Sentiment analysis not implemented."

# Text-to-Speech conversion
def text_to_speech(text, lang='en'):
    tts = gTTS(text, lang=lang)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Streamlit app
st.set_page_config(layout="wide")
st.title("Advanced Document Summarization and Analysis App")

uploaded_file = st.file_uploader("Upload your document (PDF, TXT, etc.)", type=["pdf", "txt"])

if uploaded_file is not None:
    cols, col2 = st.columns(2)
    
    if st.button("Summarize"):
        filepath = f"temp/{uploaded_file.name}"
        
        # Save uploaded file temporarily
        os.makedirs("temp", exist_ok=True)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        with cols:
            st.info("Uploaded File")
            
            # Display PDF or TXT content in iframe
            if uploaded_file.type == "application/pdf":
                base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.text(uploaded_file.read().decode("utf-8"))
        
        with col2:
            # Process the file and generate summary
            input_text = preprocess_file(filepath)
            summary = summarize_text(input_text)
            st.info("Summarization complete")
            st.success(summary)
            
            # Provide an option to save the summarized text
            st.download_button(
                label="Download Summarized Text",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            
            # Multilingual Text-to-Speech
            language = st.selectbox("Select Language", ["en", "es", "fr", "de", "it"])
            if st.button("Speak Summary"):
                audio_file = text_to_speech(summary, lang=language)
                st.audio(audio_file, format="audio/mp3")
            
            # Placeholder for sentiment analysis
            sentiment = sentiment_analysis(input_text)
            st.write("Sentiment Analysis:")
            st.write(sentiment)
        
        # Clean up temporary files
        os.remove(filepath)
