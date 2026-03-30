# 01_preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download NLTK data
print("📥 Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
print("✅ Done")

print("="*60)
print("🔧 STEP 1: TEXT PREPROCESSING")
print("="*60)

class IMDbPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Add movie-specific words to remove
        self.stop_words.update(['movie', 'film', 'one', 'get', 'would', 'br'])
        
    def clean_text(self, text):
        """Clean a single review"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and short words
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Apply stemming
        words = [self.stemmer.stem(w) for w in words]
        
        # Join back
        return ' '.join(words)
    
    def preprocess_dataset(self):
        """Preprocess entire dataset"""
        # Use the new 1000-review dataset
        input_file = 'data/imdb_1000.csv'
        
        # Load data
        df = pd.read_csv(input_file)
        print(f"📂 Loaded {len(df)} reviews from {input_file}")
        
        # Apply cleaning
        print("🔄 Cleaning reviews...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Show examples
        print("\n📝 Preprocessing examples:")
        for i in range(min(3, len(df))):
            print(f"\n{i+1}. ORIGINAL: {df['review'].iloc[i][:100]}...")
            print(f"   CLEANED:  {df['cleaned_review'].iloc[i][:100]}...")
        
        # Save processed data
        output_file = 'output/cleaned_reviews.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to {output_file}")
        
        # Calculate statistics
        df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))
        print(f"\n📊 Statistics:")
        print(f"   Avg words after cleaning: {df['word_count'].mean():.1f}")
        print(f"   Min words: {df['word_count'].min()}")
        print(f"   Max words: {df['word_count'].max()}")
        
        return df

# Run preprocessing
if __name__ == "__main__":
    preprocessor = IMDbPreprocessor()
    df = preprocessor.preprocess_dataset()
