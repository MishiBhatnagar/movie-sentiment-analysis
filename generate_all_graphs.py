# generate_all_graphs.py - Complete Visualization for Report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
import os
from wordcloud import WordCloud

print("="*70)
print("📊 GENERATING ALL GRAPHS FOR REPORT")
print("="*70)

# Create figures folder
os.makedirs('output/figures', exist_ok=True)

# Load data
df = pd.read_csv('output/cleaned_reviews.csv')
X = np.load('output/tfidf_features.npy')
y = np.load('output/labels.npy')

print(f"📂 Loaded {len(df)} reviews")

# ============================================
# FIGURE 1: DATASET ANALYSIS (3-in-1)
# ============================================
print("\n📊 Generating Figure 1: Dataset Analysis...")

fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1.1 Sentiment Distribution Bar Chart
colors = ['#2ecc71', '#e74c3c']
df['sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=colors)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for i, v in enumerate(df['sentiment'].value_counts().values):
    axes[0].text(i, v + 5, str(v), ha='center', fontsize=12)

# 1.2 Word Count Distribution
df['word_count'] = df['cleaned_review'].str.split().str.len()
for sentiment, color in zip(['positive', 'negative'], ['green', 'red']):
    data = df[df['sentiment'] == sentiment]['word_count']
    axes[1].hist(data, alpha=0.7, label=sentiment.capitalize(), 
                 color=color, bins=20, edgecolor='black')
axes[1].set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Words', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend()

# 1.3 Pie Chart
sentiment_counts = df['sentiment'].value_counts()
axes[2].pie(sentiment_counts.values, 
            labels=['Positive', 'Negative'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0))
axes[2].set_title('Sentiment Ratio', fontsize=14, fontweight='bold')

plt.suptitle('Figure 1: Movie Review Dataset Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/figure1_dataset_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure1_dataset_analysis.png")

# ============================================
# FIGURE 2: WORD CLOUDS
# ============================================
print("\n☁️ Generating Figure 2: Word Clouds...")

try:
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Positive reviews word cloud
    positive_text = ' '.join(df[df['sentiment']=='positive']['cleaned_review'])
    wordcloud_pos = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='Greens',
                             max_words=100).generate(positive_text)
    
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews - Common Words', fontsize=14, fontweight='bold')
    
    # Negative reviews word cloud
    negative_text = ' '.join(df[df['sentiment']=='negative']['cleaned_review'])
    wordcloud_neg = WordCloud(width=800, height=400,
                             background_color='white',
                             colormap='Reds',
                             max_words=100).generate(negative_text)
    
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews - Common Words', fontsize=14, fontweight='bold')
    
    plt.suptitle('Figure 2: Word Clouds by Sentiment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/figures/figure2_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: figure2_wordclouds.png")
except:
    print("⚠️ WordCloud not installed. Run: pip install wordcloud")

# ============================================
# FIGURE 3: CONFUSION MATRICES (3 models)
# ============================================
print("\n📊 Generating Figure 3: Confusion Matrices...")

# Load models
models = {}
model_files = ['naive_bayes.pkl', 'svm.pkl', 'logistic_regression.pkl']
model_names = ['Naive Bayes', 'SVM', 'Logistic Regression']

for name, file in zip(model_names, model_files):
    try:
        with open(f'output/{file}', 'rb') as f:
            import pickle
            models[name] = pickle.load(f)
        print(f"✅ Loaded {name}")
    except:
        print(f"⚠️ Could not load {file}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=axes[idx],
               annot_kws={'size': 14})
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.2%}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)

plt.suptitle('Figure 3: Confusion Matrices Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/figure3_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure3_confusion_matrices.png")

# ============================================
# FIGURE 4: ROC CURVES
# ============================================
print("\n📈 Generating Figure 4: ROC Curves...")

fig4, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2%})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Figure 4: ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/figures/figure4_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure4_roc_curves.png")

# ============================================
# FIGURE 5: LEARNING CURVES
# ============================================
print("\n📈 Generating Figure 5: Learning Curves...")

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, ax):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation Score')
    ax.set_title(title)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

fig5, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    plot_learning_curve(model, name, axes[idx])

plt.suptitle('Figure 5: Learning Curves - Training Progress', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/figure5_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure5_learning_curves.png")

# ============================================
# FIGURE 6: METRICS COMPARISON (Bar Chart)
# ============================================
print("\n📊 Generating Figure 6: Metrics Comparison...")

# Calculate metrics for each model
metrics_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    metrics_data.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

metrics_df = pd.DataFrame(metrics_data)

fig6, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics_df['Model']))
width = 0.2

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for i, metric in enumerate(metrics):
    values = metrics_df[metric].values
    bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i])
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2%}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 6: Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_df['Model'])
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/figures/figure6_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure6_metrics_comparison.png")

# ============================================
# FIGURE 7: TRAINING SIZE COMPARISON (50%,60%,70%,80%)
# ============================================
print("\n📊 Generating Figure 7: Training Size Comparison...")

train_sizes = [50, 60, 70, 80]
results = []

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size/100, random_state=42, stratify=y
    )
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Train_%': train_size,
            'Accuracy': accuracy
        })

results_df = pd.DataFrame(results)

fig7, ax = plt.subplots(figsize=(10, 6))

for name in models.keys():
    data = results_df[results_df['Model'] == name]
    ax.plot(data['Train_%'], data['Accuracy'], marker='o', linewidth=2, label=name)

ax.set_xlabel('Training Percentage (%)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Figure 7: Accuracy vs Training Size', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('output/figures/figure7_training_size.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure7_training_size.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ ALL GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print("\n📁 Figures saved in output/figures/:")
print("   ├── figure1_dataset_analysis.png")
print("   ├── figure2_wordclouds.png")
print("   ├── figure3_confusion_matrices.png")
print("   ├── figure4_roc_curves.png")
print("   ├── figure5_learning_curves.png")
print("   ├── figure6_metrics_comparison.png")
print("   └── figure7_training_size.png")
