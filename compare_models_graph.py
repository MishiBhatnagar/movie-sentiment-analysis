import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('output/model_comparison.csv')

print("="*60)
print("📊 MODEL COMPARISON RESULTS")
print("="*60)
print(df.to_string(index=False))

# Create comparison graph
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar Chart
ax1 = axes[0]
x = np.arange(len(df['Model']))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics):
    bars = ax1.bar(x + i*width, df[metric], width, label=metric, color=colors[i])
    for bar, val in zip(bars, df[metric]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels(df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

# Line Chart
ax2 = axes[1]
for metric in metrics:
    ax2.plot(df['Model'], df[metric], marker='o', linewidth=2, label=metric)

ax2.set_xlabel('Models', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Model Performance Trends', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_ylim(0.7, 0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/figures/model_comparison_distinct.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✅ Graph saved to output/figures/model_comparison_distinct.png")
