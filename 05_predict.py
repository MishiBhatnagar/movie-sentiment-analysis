# 05_predict.py
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

print("="*60)
print("🎯 STEP 5: TEST WITH NEW REVIEWS")
print("="*60)

class ReviewPredictor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['movie', 'film', 'one', 'get', 'would', 'br'])
        
        # Positive and negative word lists
        self.positive_words = ['amaz', 'great', 'excel', 'best', 'love', 'wonder', 
                              'fantast', 'brilliant', 'masterpiec', 'perfect', 
                              'outstand', 'awesome', 'beautiful', 'superb', 'enjoy']
        
        self.negative_words = ['terribl', 'worst', 'bad', 'aw', 'bore', 'wast', 
                              'disappoint', 'poor', 'awf', 'horribl', 'garbag',
                              'stupid', 'dumb', 'pathetic', 'lame', 'waste']
        
    def preprocess(self, text):
        """Preprocess a single review"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'<.*?>', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        words = [self.stemmer.stem(w) for w in words]
        return ' '.join(words)
    
    def predict(self, review_text):
        """Predict sentiment for a review"""
        # Preprocess
        cleaned = self.preprocess(review_text)
        words = cleaned.split()
        
        # Count positive and negative words
        pos_count = sum(1 for w in words if any(p in w for p in self.positive_words))
        neg_count = sum(1 for w in words if any(n in w for n in self.negative_words))
        
        # Determine sentiment
        if pos_count > neg_count:
            sentiment = "POSITIVE 😊"
            confidence = pos_count / (pos_count + neg_count + 1)
        elif neg_count > pos_count:
            sentiment = "NEGATIVE 😠"
            confidence = neg_count / (pos_count + neg_count + 1)
        else:
            sentiment = "NEUTRAL 😐"
            confidence = 0.5
        
        # Display result
        print("\n" + "-"*60)
        print(f"📝 Review: {review_text[:150]}..." if len(review_text) > 150 else f"📝 Review: {review_text}")
        print(f"🧹 Cleaned: {cleaned[:100]}..." if len(cleaned) > 100 else f"🧹 Cleaned: {cleaned}")
        print(f"\n🎯 Prediction: {sentiment}")
        print(f"📊 Confidence: {confidence:.1%}")
        print(f"   Positive words found: {pos_count}")
        print(f"   Negative words found: {neg_count}")
        
        # Show matched words
        pos_matches = [w for w in words if any(p in w for p in self.positive_words)]
        neg_matches = [w for w in words if any(n in w for n in self.negative_words)]
        
        if pos_matches:
            print(f"   😊 Positive matches: {', '.join(pos_matches[:5])}")
        if neg_matches:
            print(f"   😠 Negative matches: {', '.join(neg_matches[:5])}")
        
        return sentiment, confidence

# Test with sample reviews
if __name__ == "__main__":
    predictor = ReviewPredictor()
    
    # Test reviews
    test_reviews = [
        "This movie was absolutely amazing! The acting was superb and I loved every minute.",
        "Terrible film, complete waste of time. Boring and predictable plot.",
        "The movie was okay, nothing special but not terrible either.",
        "Best movie ever! Outstanding performances and brilliant direction.",
        "Awful acting and poor screenplay. Disappointing experience."
    ]
    
    print("\n🔮 Testing with sample reviews:")
    for review in test_reviews:
        predictor.predict(review)
    
    # Try with actual reviews from dataset
    try:
        df = pd.read_csv('output/cleaned_reviews.csv')
        print("\n\n📊 Testing with actual reviews from dataset:")
        for i in range(min(3, len(df))):
            print(f"\nOriginal ({df['sentiment'].iloc[i]}):")
            predictor.predict(df['review'].iloc[i])
    except:
        print("\n⚠️ Could not load dataset for testing")
