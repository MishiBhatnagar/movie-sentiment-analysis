# 02_feature_extraction.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os

print("="*60)
print("🔍 STEP 2: FEATURE EXTRACTION")
print("="*60)

class FeatureExtractor:
    def __init__(self, max_features=1000):  # Increased from 500 to 1000
        self.max_features = max_features
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,  # Increased from 2 to 5 (with more data)
            max_df=0.8,  # Slightly reduced
            ngram_range=(1, 2)
        )
        
    def extract_bow(self, reviews):
        """Extract Bag of Words features"""
        print("\n📊 Extracting Bag of Words features...")
        X = self.bow_vectorizer.fit_transform(reviews)
        feature_names = self.bow_vectorizer.get_feature_names_out()
        
        print(f"   Shape: {X.shape}")
        print(f"   Features: {X.shape[1]} words")
        print(f"   Sample features: {list(feature_names[:10])}")
        
        return X, feature_names
    
    def extract_tfidf(self, reviews):
        """Extract TF-IDF features"""
        print("\n📊 Extracting TF-IDF features...")
        X = self.tfidf_vectorizer.fit_transform(reviews)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"   Shape: {X.shape}")
        print(f"   Features: {X.shape[1]} words")
        print(f"   Sample features: {list(feature_names[:10])}")
        
        return X, feature_names
    
    def save_features(self, X, filename):
        """Save features to file"""
        np.save(f'output/{filename}.npy', X.toarray())
        print(f"✅ Saved to output/{filename}.npy")

# Load cleaned reviews
df = pd.read_csv('output/cleaned_reviews.csv')
reviews = df['cleaned_review'].tolist()
labels = (df['sentiment'] == 'positive').astype(int).values

print(f"📂 Loaded {len(reviews)} cleaned reviews")
print(f"   Positive: {sum(labels)}")
print(f"   Negative: {len(labels)-sum(labels)}")

# Extract features
extractor = FeatureExtractor(max_features=1000)

# BoW features
X_bow, bow_features = extractor.extract_bow(reviews)
extractor.save_features(X_bow, 'bow_features')

# TF-IDF features (better for ML)
X_tfidf, tfidf_features = extractor.extract_tfidf(reviews)
extractor.save_features(X_tfidf, 'tfidf_features')

# Save labels
np.save('output/labels.npy', labels)
print(f"\n✅ Labels saved to output/labels.npy")

# Show top words
word_counts = np.array(X_bow.sum(axis=0)).flatten()
top_indices = np.argsort(word_counts)[-20:]
print(f"\n📝 Top 20 most frequent words:")
for idx in reversed(top_indices):
    print(f"   {bow_features[idx]}: {int(word_counts[idx])}")
