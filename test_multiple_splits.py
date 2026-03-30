# test_multiple_splits.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

print("="*70)
print("📊 TESTING WITH DIFFERENT TRAINING PERCENTAGES")
print("="*70)

# Load data
X = np.load('output/tfidf_features.npy')
y = np.load('output/labels.npy')

print(f"Total samples: {len(y)}")
print(f"Features: {X.shape[1]}")

# Models
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'SVM': SVC(kernel='linear', C=1.0),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000)
}

# Test different training percentages
train_percentages = [50, 60, 70, 80]
results = []

for train_size in train_percentages:
    print(f"\n{'-'*50}")
    print(f"📈 TRAINING SIZE: {train_size}%")
    print(f"{'-'*50}")
    
    test_size = 100 - train_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size/100, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate confusion matrix for additional metrics
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate all metrics as per assignment
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
            
            # MCC
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numerator / denominator if denominator > 0 else 0
        else:
            specificity = sensitivity = fpr = fnr = npv = fdr = mcc = 0
        
        results.append({
            'Model': name,
            'Train_%': train_size,
            'Train_samples': len(X_train),
            'Test_samples': len(X_test),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'FPR': fpr,
            'FNR': fnr,
            'NPV': npv,
            'FDR': fdr,
            'MCC': mcc
        })
        
        print(f"\n{name}:")
        print(f"   Accuracy:  {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall:    {recall:.2%}")
        print(f"   F1:        {f1:.2%}")
        print(f"   MCC:       {mcc:.3f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('output/multiple_splits_results.csv', index=False)
print(f"\n✅ Results saved to output/multiple_splits_results.csv")

# Display summary table
print("\n" + "="*70)
print("📊 SUMMARY TABLE - ALL METRICS")
print("="*70)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_df.to_string(index=False))

# Create the graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# FIGURE 9: Accuracy and F1-Score
ax1 = axes[0, 0]
for model in models.keys():
    model_data = results_df[results_df['Model'] == model]
    ax1.plot(model_data['Train_%'], model_data['Accuracy'], marker='o', linewidth=2, label=f'{model} - Accuracy')
    ax1.plot(model_data['Train_%'], model_data['F1'], marker='s', linestyle='--', linewidth=2, label=f'{model} - F1')
ax1.set_xlabel('Training Percentage (%)', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Figure 9: Accuracy and F1-Score vs Training Size', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# FIGURE 10: Error Rates
ax2 = axes[0, 1]
for model in models.keys():
    model_data = results_df[results_df['Model'] == model]
    ax2.plot(model_data['Train_%'], model_data['FDR'], marker='o', linewidth=2, label=f'{model} - FDR')
    ax2.plot(model_data['Train_%'], model_data['FNR'], marker='s', linewidth=2, label=f'{model} - FNR')
    ax2.plot(model_data['Train_%'], model_data['FPR'], marker='^', linewidth=2, label=f'{model} - FPR')
ax2.set_xlabel('Training Percentage (%)', fontsize=12)
ax2.set_ylabel('Rate', fontsize=12)
ax2.set_title('Figure 10: Error Rates vs Training Size', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.1)

# FIGURE 11: Precision and MCC
ax3 = axes[1, 0]
for model in models.keys():
    model_data = results_df[results_df['Model'] == model]
    ax3.plot(model_data['Train_%'], model_data['Precision'], marker='o', linewidth=2, label=f'{model} - Precision')
    ax3.plot(model_data['Train_%'], model_data['MCC'], marker='s', linewidth=2, label=f'{model} - MCC')
ax3.set_xlabel('Training Percentage (%)', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Figure 11: Precision and MCC vs Training Size', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.1)

# Additional: Sensitivity and Specificity
ax4 = axes[1, 1]
for model in models.keys():
    model_data = results_df[results_df['Model'] == model]
    ax4.plot(model_data['Train_%'], model_data['Sensitivity'], marker='o', linewidth=2, label=f'{model} - Sensitivity')
    ax4.plot(model_data['Train_%'], model_data['Specificity'], marker='s', linewidth=2, label=f'{model} - Specificity')
ax4.set_xlabel('Training Percentage (%)', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Sensitivity and Specificity vs Training Size', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.1)

plt.suptitle('Movie Sentiment Analysis - Model Performance Across Training Sizes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/performance_vs_training_size.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Graph saved to output/figures/performance_vs_training_size.png")
