# 03_train_models_cv.py - Using Cross-Validation for small dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("="*60)
print("🤖 TRAINING WITH CROSS-VALIDATION (5-Fold)")
print("="*60)

# Load data
X = np.load('output/tfidf_features.npy')
y = np.load('output/labels.npy')

print(f"\n📊 Total samples: {len(y)}")
print(f"   Features: {X.shape[1]}")

# Define models
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'SVM': SVC(kernel='linear', C=1.0),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000)
}

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n📊 5-Fold Cross-Validation Results:")
print("-" * 60)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'all_scores': scores
    }
    
    print(f"\n{name}:")
    print(f"   Fold accuracies: {[f'{s:.2%}' for s in scores]}")
    print(f"   Mean Accuracy:   {scores.mean():.2%} ± {scores.std():.2%}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['mean'])
print("\n" + "="*60)
print(f"✅ BEST MODEL: {best_model[0]} with {best_model[1]['mean']:.2%} accuracy")
print("="*60)

# Save results
results_df = pd.DataFrame([
    {'Model': name, 'Mean_Accuracy': res['mean'], 'Std': res['std']}
    for name, res in results.items()
])
results_df.to_csv('output/cv_results.csv', index=False)
print("\n✅ Cross-validation results saved to output/cv_results.csv")
