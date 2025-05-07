import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

def download_nltk_resources():
    """Ensure all required NLTK resources are available"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        if not nltk.data.find('tokenizers/punkt'):
            raise Exception("Punkt tokenizer not found")
    except Exception as e:
        print(f"\nNLTK resource download failed: {e}")
        print("Trying alternative download method...")
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('all', quiet=True)
        except Exception as alt_e:
            print(f"Alternative download failed: {alt_e}")
            raise Exception("Could not download required NLTK resources")

def clean_text(text):
    """Robust text cleaning with comprehensive error handling"""
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
            
        lemmatizer = nltk.stem.WordNetLemmatizer()
        stop_words = set(nltk.corpus.stopwords.words('english'))
        
        text = text.lower().strip()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\d+', '[NUMBER]', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = nltk.word_tokenize(text)
        except:
            tokens = text.split()
            
        clean_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in stop_words:
                try:
                    clean_tokens.append(lemmatizer.lemmatize(token))
                except:
                    clean_tokens.append(token)
        
        return ' '.join(clean_tokens) if clean_tokens else ""
    except Exception as e:
        print(f"Warning: Error cleaning text - {str(e)[:100]}...")
        return ""

def train_and_save_model():
    """Main training function with comprehensive validation"""
    try:
        print("\nInitializing spam detection model training...")
        
        if not os.path.exists("mail_data.csv"):
            raise FileNotFoundError("mail_data.csv not found in current directory")
        
        df = pd.read_csv("mail_data.csv", encoding='latin-1')
        
        if not {'Category', 'Message'}.issubset(df.columns):
            raise ValueError("CSV must contain 'Category' and 'Message' columns")
            
        df = df[['Category', 'Message']].copy()
        df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        print("\nCleaning text data...")
        df['cleaned_text'] = df['Message'].apply(clean_text)
        df = df[df['cleaned_text'].str.len() > 0]
        
        if len(df) == 0:
            raise ValueError("All texts were filtered out during cleaning")
        
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            min_df=2,
            stop_words='english'
        )
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['Category'].values
        
        if X.shape[1] == 0:
            raise ValueError("No features remaining after vectorization")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\nBuilding neural network...")
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("\nTraining model (this may take a while)...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=64,
            callbacks=[EarlyStopping(patience=3)],
            verbose=1
        )
        
        model.save('spam_model.keras')
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        print("\nTraining successfully completed!")
        print("Saved models:")
        print(f"- spam_model.keras ({(os.path.getsize('spam_model.keras')/1024):.1f} KB)")
        print(f"- tfidf_vectorizer.pkl ({(os.path.getsize('tfidf_vectorizer.pkl')/1024):.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure mail_data.csv has valid data with 'Category' and 'Message' columns")
        print("2. Check your messages contain more than just stopwords")
        print("3. Try running as administrator if permission errors occur")
        return False

if __name__ == "__main__":
    download_nltk_resources()
    success = train_and_save_model()
    
    if not success:
        print("\nTraining failed. Please address the issues above and try again.")
        exit(1)