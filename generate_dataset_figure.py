# generate_dataset_figure.py - Figure 1: Dataset Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

print("="*70)
print("📊 GENERATING FIGURE 1: MOVIE REVIEW DATASET ANALYSIS")
print("="*70)

# Create figures folder
os.makedirs('output/figures', exist_ok=True)

# Load your cleaned data
df = pd.read_csv('output/cleaned_reviews.csv')
print(f"📂 Loaded {len(df)} reviews")

# Calculate statistics
df['review_length'] = df['review'].apply(len)
df['word_count_original'] = df['review'].apply(lambda x: len(str(x).split()))
df['word_count_cleaned'] = df['cleaned_review'].apply(lambda x: len(str(x).split()))

# ============================================
# FIGURE 1: COMPREHENSIVE DATASET ANALYSIS (4-in-1)
# ============================================
print("\n📊 Creating Figure 1: Dataset Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 SENTIMENT DISTRIBUTION (Top Left)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('(a) Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment', fontsize=12)
axes[0, 0].set_ylabel('Number of Reviews', fontsize=12)

# Add value labels on bars
for bar, val in zip(bars, sentiment_counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{val}\n({val/len(df)*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# 1.2 WORD COUNT DISTRIBUTION BY SENTIMENT (Top Right)
for sentiment, color in zip(['positive', 'negative'], ['#2ecc71', '#e74c3c']):
    data = df[df['sentiment'] == sentiment]['word_count_original']
    axes[0, 1].hist(data, alpha=0.7, label=sentiment.capitalize(), 
                     color=color, bins=30, edgecolor='black', linewidth=0.5)
axes[0, 1].set_title('(b) Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Words per Review', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].legend()
axes[0, 1].axvline(df['word_count_original'].mean(), color='blue', linestyle='--', 
                   linewidth=1.5, label=f'Mean: {df["word_count_original"].mean():.0f}')
axes[0, 1].legend()

# 1.3 PIE CHART - SENTIMENT RATIO (Bottom Left)
axes[1, 0].pie(sentiment_counts.values, 
               labels=['Positive', 'Negative'],
               colors=['#2ecc71', '#e74c3c'],
               autopct='%1.1f%%',
               startangle=90,
               explode=(0.05, 0),
               shadow=True,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1, 0].set_title('(c) Sentiment Ratio', fontsize=14, fontweight='bold')

# 1.4 REVIEW LENGTH STATISTICS (Bottom Right)
stats_data = {
    'Positive': {
        'Mean': df[df['sentiment']=='positive']['word_count_original'].mean(),
        'Min': df[df['sentiment']=='positive']['word_count_original'].min(),
        'Max': df[df['sentiment']=='positive']['word_count_original'].max()
    },
    'Negative': {
        'Mean': df[df['sentiment']=='negative']['word_count_original'].mean(),
        'Min': df[df['sentiment']=='negative']['word_count_original'].min(),
        'Max': df[df['sentiment']=='negative']['word_count_original'].max()
    }
}

# Create a table
categories = ['Positive', 'Negative']
means = [stats_data['Positive']['Mean'], stats_data['Negative']['Mean']]
mins = [stats_data['Positive']['Min'], stats_data['Negative']['Min']]
maxs = [stats_data['Positive']['Max'], stats_data['Negative']['Max']]

x_pos = np.arange(len(categories))
width = 0.25

bars1 = axes[1, 1].bar(x_pos - width, means, width, label='Mean', color='#3498db', edgecolor='black')
bars2 = axes[1, 1].bar(x_pos, mins, width, label='Minimum', color='#f39c12', edgecolor='black')
bars3 = axes[1, 1].bar(x_pos + width, maxs, width, label='Maximum', color='#e74c3c', edgecolor='black')

axes[1, 1].set_title('(d) Review Length Statistics by Sentiment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Sentiment', fontsize=12)
axes[1, 1].set_ylabel('Number of Words', fontsize=12)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 2,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Figure 1: Movie Review Dataset Analysis\n(24 Original Reviews from IMDb)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/figures/figure1_dataset_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: figure1_dataset_analysis.png")

# ============================================
# FIGURE 1B: ADDITIONAL - WORD CLOUDS (Optional)
# ============================================
print("\n☁️ Generating Word Clouds for Positive and Negative Reviews...")

try:
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Positive reviews word cloud
    positive_text = ' '.join(df[df['sentiment']=='positive']['cleaned_review'].tolist())
    wordcloud_pos = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='Greens',
                             max_words=100,
                             contour_width=1,
                             contour_color='darkgreen').generate(positive_text)
    
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews - Most Common Words', fontsize=14, fontweight='bold')
    
    # Negative reviews word cloud
    negative_text = ' '.join(df[df['sentiment']=='negative']['cleaned_review'].tolist())
    wordcloud_neg = WordCloud(width=800, height=400,
                             background_color='white',
                             colormap='Reds',
                             max_words=100,
                             contour_width=1,
                             contour_color='darkred').generate(negative_text)
    
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews - Most Common Words', fontsize=14, fontweight='bold')
    
    plt.suptitle('Figure 1b: Word Clouds by Sentiment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/figures/figure1b_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: figure1b_wordclouds.png")
except Exception as e:
    print(f"⚠️ WordCloud not available: {e}")
    print("   Run: pip install wordcloud")

# ============================================
# PRINT DATASET STATISTICS FOR REPORT
# ============================================
print("\n" + "="*70)
print("📊 DATASET STATISTICS FOR YOUR REPORT")
print("="*70)

print(f"""
| Feature | Value |
|---------|-------|
| Total Reviews | {len(df)} |
| Positive Reviews | {sum(df['sentiment']=='positive')} ({sum(df['sentiment']=='positive')/len(df)*100:.1f}%) |
| Negative Reviews | {sum(df['sentiment']=='negative')} ({sum(df['sentiment']=='negative')/len(df)*100:.1f}%) |
| Average Review Length | {df['word_count_original'].mean():.1f} words |
| Shortest Review | {df['word_count_original'].min()} words |
| Longest Review | {df['word_count_original'].max()} words |
| Average Cleaned Length | {df['word_count_cleaned'].mean():.1f} words |
| Vocabulary Size (after cleaning) | {len(set(' '.join(df['cleaned_review']).split()))} unique words |
""")

print("\n📊 Most Common Words in Positive Reviews:")
pos_words = ' '.join(df[df['sentiment']=='positive']['cleaned_review']).split()
from collections import Counter
pos_top = Counter(pos_words).most_common(10)
for word, count in pos_top:
    print(f"   {word}: {count}")

print("\n📊 Most Common Words in Negative Reviews:")
neg_words = ' '.join(df[df['sentiment']=='negative']['cleaned_review']).split()
neg_top = Counter(neg_words).most_common(10)
for word, count in neg_top:
    print(f"   {word}: {count}")

print("\n" + "="*70)
print("✅ DATASET ANALYSIS COMPLETE!")
print("="*70)
