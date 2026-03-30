# 03_train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

print("="*60)
print("🤖 STEP 3: TRAINING MODELS WITH 1000 REVIEWS")
print("="*60)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load features and labels"""
        X = np.load('output/tfidf_features.npy')
        y = np.load('output/labels.npy')
        
        print(f"📂 Loaded features: {X.shape}")
        print(f"   Positive samples: {sum(y)}")
        print(f"   Negative samples: {len(y)-sum(y)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.3):
        """Split into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split (70-30):")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing:  {X_test.shape[0]} samples")
        print(f"   Training distribution: {np.bincount(y_train)}")
        print(f"   Testing distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes"""
        print("\n🔹 Training Naive Bayes...")
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        self.models['Naive Bayes'] = model
        print("   ✅ Done")
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM"""
        print("\n🔹 Training SVM...")
        model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        print("   ✅ Done")
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression"""
        print("\n🔹 Training Logistic Regression...")
        model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear',
                                  random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("   ✅ Done")
        return model
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            # Print results
            print(f"\n{name}:")
            print(f"   Accuracy: {accuracy:.2%}")
            if cm.shape == (2,2):
                print(f"   Confusion Matrix:")
                print(f"               Predicted")
                print(f"               Neg  Pos")
                print(f"   Actual Neg  {cm[0,0]:3d}  {cm[0,1]:3d}")
                print(f"          Pos  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            filename = f'output/{name.lower().replace(" ", "_")}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved {filename}")
    
    def save_results(self):
        """Save results to CSV"""
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': f"{res['accuracy']:.2%}",
                'TN': res['confusion_matrix'][0,0] if res['confusion_matrix'].shape==(2,2) else 0,
                'FP': res['confusion_matrix'][0,1] if res['confusion_matrix'].shape==(2,2) else 0,
                'FN': res['confusion_matrix'][1,0] if res['confusion_matrix'].shape==(2,2) else 0,
                'TP': res['confusion_matrix'][1,1] if res['confusion_matrix'].shape==(2,2) else 0
            }
            for name, res in self.results.items()
        ])
        results_df.to_csv('output/results.csv', index=False)
        print(f"\n✅ Results saved to output/results.csv")
        print(results_df.to_string(index=False))

# Run training
if __name__ == "__main__":
    # Create trainer
    trainer = ModelTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.3)
    
    # Train models
    trainer.train_naive_bayes(X_train, y_train)
    trainer.train_svm(X_train, y_train)
    trainer.train_logistic_regression(X_train, y_train)
    
    # Evaluate
    trainer.evaluate(X_test, y_test)
    
    # Save models and results
    trainer.save_models()
    trainer.save_results()
