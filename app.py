import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Must be the first Streamlit command
st.set_page_config(
    page_title="Spam Email Detector", 
    layout="wide",
    page_icon="📧"
)

# Initialize NLP tools with error handling
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error loading NLTK resources: {e}")
        return False

if not load_nltk_resources():
    st.stop()

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('spam_model.keras')
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure you've run train_model.py first to create the model files")
        st.stop()

model, vectorizer = load_models()

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Enhanced text cleaning function
def clean_text(text):
    try:
        if not text or not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\d+', '[NUMBER]', text)
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization with fallback
        try:
            tokens = nltk.word_tokenize(text)
        except:
            tokens = text.split()
            
        # Lemmatization and filtering
        clean_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in stop_words:
                try:
                    clean_tokens.append(lemmatizer.lemmatize(token))
                except:
                    clean_tokens.append(token)
        
        return ' '.join(clean_tokens) if clean_tokens else ""
    except Exception as e:
        st.warning(f"Text cleaning warning: {str(e)[:100]}...")
        return ""

# Main interface
st.title("📧 Spam Email Detection System")
st.markdown("""
<style>
.stTextArea textarea {
    height: 200px;
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
}
.stButton button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    email_input = st.text_area(
        "Paste the email content here:", 
        height=250,
        placeholder="Enter or paste email content to analyze for spam..."
    )

    if st.button("Check for Spam", type="primary"):
        if email_input:
            with st.spinner("Analyzing email content..."):
                # Clean and predict
                cleaned_text = clean_text(email_input)
                try:
                    vectorized = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vectorized, verbose=0)[0][0]
                    
                    # Display results
                    st.subheader("Results")
                    if prediction > 0.5:
                        st.error(f"🚨 SPAM DETECTED (confidence: {prediction*100:.1f}%)")
                        st.warning("**Warning:** This email appears suspicious. Be cautious with:")
                        st.markdown("- Links or attachments\n- Requests for personal information\n- Urgent demands for action")
                    else:
                        st.success(f"✅ LEGITIMATE EMAIL (confidence: {(1-prediction)*100:.1f}%)")
                        st.info("This email appears to be safe for normal use.")
                    
                    # Show processed text
                    with st.expander("View processed text analysis"):
                        st.code(cleaned_text)
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("⚠️ Please enter some email content to analyze")

with col2:
    st.markdown("""
    ### 📝 How It Works
    1. Paste email content in the box
    2. Click "Check for Spam"
    3. View the analysis results
    
    ### 🔍 Detection Features
    - Suspicious keywords
    - URL patterns
    - Email structure
    - Common spam indicators
    
    ### ⚠️ Spam Indicators
    - Urgent requests
    - "You've won" messages
    - Misspelled words
    - Suspicious links
    """)

# Footer
st.markdown("---")
st.markdown("""
*Note: This is an AI model and may not be 100% accurate. Always use caution with suspicious emails.*
""")