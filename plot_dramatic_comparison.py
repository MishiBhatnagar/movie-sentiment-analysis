import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('output/model_comparison_dramatic.csv')

print("="*60)
print("📊 DRAMATIC MODEL COMPARISON")
print("="*60)
print(df.to_string(index=False))

# Create impressive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar Chart with clear differences
ax1 = axes[0, 0]
x = np.arange(len(df['Model']))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    bars = ax1.bar(x + i*width, df[metric], width, label=metric, color=colors[i], alpha=0.8)
    for bar, val in zip(bars, df[metric]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Model Performance Comparison\nClear Performance Gap', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels(['Simple BoW\n(Naive Bayes)', 'Advanced TF-IDF\n(SVM)', 'Character n-grams\n(Gradient Boosting)'])
ax1.legend(loc='lower right')
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Line Chart showing progression
ax2 = axes[0, 1]
for metric in metrics:
    ax2.plot(df['Model'], df[metric], marker='o', linewidth=2, markersize=8, label=metric)

ax2.set_xlabel('Models', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Performance Progression\nIncreasing Complexity', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_ylim(0.6, 0.95)
ax2.grid(True, alpha=0.3)

# 3. Improvement Chart
ax3 = axes[1, 0]
improvements = [
    df['Accuracy'].iloc[1] - df['Accuracy'].iloc[0],
    df['Accuracy'].iloc[2] - df['Accuracy'].iloc[1],
    df['Accuracy'].iloc[2] - df['Accuracy'].iloc[0]
]
labels = ['SVM vs NB', 'GB vs SVM', 'GB vs NB']
colors_imp = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax3.bar(labels, improvements, color=colors_imp, alpha=0.7)
ax3.set_ylabel('Accuracy Improvement', fontsize=12)
ax3.set_title('Performance Improvements\nModel to Model', fontsize=14, fontweight='bold')
for bar, val in zip(bars, improvements):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'+{val:.1%}', ha='center', va='bottom', fontsize=11)
ax3.set_ylim(0, max(improvements) + 0.05)

# 4. Confusion Matrix Summary
ax4 = axes[1, 1]
summary_data = {
    'Model 1\n(Simple)': [df['Accuracy'].iloc[0], df['F1-Score'].iloc[0]],
    'Model 2\n(Medium)': [df['Accuracy'].iloc[1], df['F1-Score'].iloc[1]],
    'Model 3\n(Advanced)': [df['Accuracy'].iloc[2], df['F1-Score'].iloc[2]]
}

x = np.arange(len(summary_data))
width = 0.35

bars1 = ax4.bar(x - width/2, [v[0] for v in summary_data.values()], width, label='Accuracy', color='#2ecc71')
bars2 = ax4.bar(x + width/2, [v[1] for v in summary_data.values()], width, label='F1-Score', color='#f39c12')

ax4.set_xlabel('Models', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Accuracy vs F1-Score\nModel Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(summary_data.keys())
ax4.legend()
ax4.set_ylim(0, 1.0)
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Dramatic Model Performance Differences', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/dramatic_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Graph saved to output/figures/dramatic_model_comparison.png")
