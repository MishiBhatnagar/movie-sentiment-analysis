# 03_train_models_distinct.py - Make each model unique
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

print("="*70)
print("🤖 TRAINING MODELS WITH DISTINCT APPROACHES")
print("="*70)

# Load cleaned reviews
df = pd.read_csv('output/cleaned_reviews.csv')
reviews = df['cleaned_review'].tolist()
labels = (df['sentiment'] == 'positive').astype(int).values

print(f"📂 Loaded {len(reviews)} reviews")
print(f"   Positive: {sum(labels)}")
print(f"   Negative: {len(labels)-sum(labels)}")

# ============================================
# METHOD 1: NAIVE BAYES with BoW (Simple counts)
# ============================================
print("\n" + "="*60)
print("📊 METHOD 1: NAIVE BAYES with Bag-of-Words")
print("="*60)

bow_vectorizer = CountVectorizer(max_features=500)
X_bow = bow_vectorizer.fit_transform(reviews)

X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, labels, test_size=0.3, random_state=42, stratify=labels
)

nb_model = MultinomialNB(alpha=0.5)  # Less smoothing = more sensitive
nb_model.fit(X_train_bow, y_train)
y_pred_nb = nb_model.predict(X_test_bow)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

print(f"\n✅ Naive Bayes (BoW) Results:")
print(f"   Accuracy:  {accuracy_nb:.2%}")
print(f"   Precision: {precision_nb:.2%}")
print(f"   Recall:    {recall_nb:.2%}")
print(f"   F1-Score:  {f1_nb:.2%}")
print(f"   Confusion Matrix:")
print(f"               Predicted")
print(f"               Neg  Pos")
print(f"   Actual Neg  {cm_nb[0,0]:4d}  {cm_nb[0,1]:4d}")
print(f"          Pos  {cm_nb[1,0]:4d}  {cm_nb[1,1]:4d}")

# ============================================
# METHOD 2: SVM with TF-IDF + Bigrams
# ============================================
print("\n" + "="*60)
print("📊 METHOD 2: SVM with TF-IDF + Bigrams")
print("="*60)

tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_tfidf = tfidf_vectorizer.fit_transform(reviews)

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, labels, test_size=0.3, random_state=42, stratify=labels
)

svm_model = SVC(kernel='linear', C=0.5)  # Different C value
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print(f"\n✅ SVM (TF-IDF + Bigrams) Results:")
print(f"   Accuracy:  {accuracy_svm:.2%}")
print(f"   Precision: {precision_svm:.2%}")
print(f"   Recall:    {recall_svm:.2%}")
print(f"   F1-Score:  {f1_svm:.2%}")
print(f"   Confusion Matrix:")
print(f"               Predicted")
print(f"               Neg  Pos")
print(f"   Actual Neg  {cm_svm[0,0]:4d}  {cm_svm[0,1]:4d}")
print(f"          Pos  {cm_svm[1,0]:4d}  {cm_svm[1,1]:4d}")

# ============================================
# METHOD 3: LOGISTIC REGRESSION with TF-IDF + L1 Regularization
# ============================================
print("\n" + "="*60)
print("📊 METHOD 3: Logistic Regression with L1 Regularization")
print("="*60)

# Use the same TF-IDF but different preprocessing
tfidf_vectorizer_lr = TfidfVectorizer(max_features=800, min_df=3)
X_tfidf_lr = tfidf_vectorizer_lr.fit_transform(reviews)

X_train_lr, X_test_lr, y_train, y_test = train_test_split(
    X_tfidf_lr, labels, test_size=0.3, random_state=42, stratify=labels
)

lr_model = LogisticRegression(C=2.0, penalty='l1', solver='liblinear', max_iter=2000)
lr_model.fit(X_train_lr, y_train)
y_pred_lr = lr_model.predict(X_test_lr)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print(f"\n✅ Logistic Regression (L1 Regularization) Results:")
print(f"   Accuracy:  {accuracy_lr:.2%}")
print(f"   Precision: {precision_lr:.2%}")
print(f"   Recall:    {recall_lr:.2%}")
print(f"   F1-Score:  {f1_lr:.2%}")
print(f"   Confusion Matrix:")
print(f"               Predicted")
print(f"               Neg  Pos")
print(f"   Actual Neg  {cm_lr[0,0]:4d}  {cm_lr[0,1]:4d}")
print(f"          Pos  {cm_lr[1,0]:4d}  {cm_lr[1,1]:4d}")

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "="*70)
print("📊 COMPARISON TABLE")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Naive Bayes (BoW)', 'SVM (TF-IDF + Bigrams)', 'Logistic Regression (L1)'],
    'Accuracy': [accuracy_nb, accuracy_svm, accuracy_lr],
    'Precision': [precision_nb, precision_svm, precision_lr],
    'Recall': [recall_nb, recall_svm, recall_lr],
    'F1-Score': [f1_nb, f1_svm, f1_lr]
})

print(comparison_df.to_string(index=False))

# Save models
print("\n📦 Saving models...")
with open('output/naive_bayes_bow.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open('output/svm_tfidf.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('output/logistic_regression_l1.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save vectorizers
with open('output/bow_vectorizer.pkl', 'wb') as f:
    pickle.dump(bow_vectorizer, f)
with open('output/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

comparison_df.to_csv('output/model_comparison.csv', index=False)
print("✅ Models and results saved!")
print("\n🎯 Now you should see CLEAR differences between models!")
