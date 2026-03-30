# 07_visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle
from sklearn.model_selection import train_test_split

print("="*60)
print("📊 STEP 7: GENERATING ALL VISUALIZATIONS")
print("="*60)

# Create figures folder
os.makedirs('output/figures', exist_ok=True)

# Load data
df = pd.read_csv('output/cleaned_reviews.csv')
X = np.load('output/tfidf_features.npy')
y = np.load('output/labels.npy')

print(f"📂 Loaded {len(df)} reviews")

# 1. SENTIMENT DISTRIBUTION (Bar Chart & Pie Chart)
print("\n📊 Generating Sentiment Distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1.1 Bar Chart
colors = ['#2ecc71', '#e74c3c']
df['sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=colors)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
for i, v in enumerate(df['sentiment'].value_counts().values):
    axes[0].text(i, v + 0.1, str(v), ha='center', fontsize=12)

# 1.2 Word Count Distribution by Sentiment
for sentiment, color in zip(['positive', 'negative'], ['green', 'red']):
    data = df[df['sentiment'] == sentiment]
    word_counts = data['cleaned_review'].str.split().str.len()
    axes[1].hist(word_counts, alpha=0.7, label=sentiment.capitalize(), 
                 color=color, bins=10, edgecolor='black')
axes[1].set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Words (after cleaning)', fontsize=12)
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

plt.suptitle('Movie Review Dataset Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/dataset_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: dataset_analysis.png")

# 2. WORD CLOUDS (if wordcloud is installed)
try:
    from wordcloud import WordCloud
    
    print("\n☁️ Generating Word Clouds...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    plt.tight_layout()
    plt.savefig('output/figures/wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: wordclouds.png")
except ImportError:
    print("⚠️ wordcloud not installed. Skipping word clouds.")
    print("   Install with: pip install wordcloud")

# 3. CONFUSION MATRICES FOR ALL MODELS
print("\n📊 Generating Confusion Matrices...")

# Load models
models = {}
model_files = ['naive_bayes.pkl', 'svm.pkl', 'logistic_regression.pkl']
model_names = ['Naive Bayes', 'SVM', 'Logistic Regression']

for name, file in zip(model_names, model_files):
    try:
        with open(f'output/{file}', 'rb') as f:
            models[name] = pickle.load(f)
        print(f"✅ Loaded {name}")
    except:
        print(f"⚠️ Could not load {file}")

# Split data for testing
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=axes[idx],
               annot_kws={'size': 14})
    
    # Calculate accuracy
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.2%}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: all_confusion_matrices.png")

# 4. METRICS COMPARISON BAR CHART
print("\n📊 Generating Metrics Comparison Chart...")

# Load detailed results
try:
    results_df = pd.read_csv('output/detailed_results.csv', index_col=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'MCC']
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (idx, model) in enumerate(results_df.iterrows()):
        values = [model[m] for m in metrics if m != 'MCC']
        if 'MCC' in metrics:
            mcc_value = model['MCC']
            values.append(mcc_value)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=idx, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val < 1.0:
                label = f'{val:.2%}' if val < 1.0 else f'{val:.3f}'
            else:
                label = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   label, ha='center', va='bottom', fontsize=9, rotation=45)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output/figures/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: all_metrics_comparison.png")
    
except Exception as e:
    print(f"⚠️ Could not generate metrics comparison: {e}")

# 5. ROC CURVES
print("\n📈 Generating ROC Curves...")

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2%})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/figures/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: roc_curves_comparison.png")

# 6. TRAINING PROGRESS (Learning Curves)
print("\n📈 Generating Learning Curves...")

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.3, 1.0, 5),
        scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    axes.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    axes.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    axes.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation Score')
    axes.set_title(title)
    axes.set_xlabel('Training Examples')
    axes.set_ylabel('Accuracy')
    axes.legend(loc='best')
    axes.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    plot_learning_curve(model, name, X, y, axes[idx])

plt.suptitle('Learning Curves - Training Progress', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: learning_curves.png")

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS COMPLETED!")
print("="*60)
print("\n📁 Figures saved in output/figures/:")
print("   - dataset_analysis.png")
print("   - wordclouds.png (if wordcloud installed)")
print("   - all_confusion_matrices.png")
print("   - all_metrics_comparison.png")
print("   - roc_curves_comparison.png")
print("   - learning_curves.png")
