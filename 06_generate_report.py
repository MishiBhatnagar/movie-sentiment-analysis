# 06_generate_report.py
import pandas as pd
import numpy as np
import os

print("="*60)
print("📝 STEP 6: GENERATING REPORT TABLES")
print("="*60)

# Load data
try:
    df = pd.read_csv('output/cleaned_reviews.csv')
    print("✅ Loaded cleaned reviews")
except:
    print("⚠️ Could not load cleaned reviews")
    df = pd.DataFrame()

try:
    results = pd.read_csv('output/detailed_results.csv', index_col=0)
    print("✅ Loaded detailed results")
except:
    print("⚠️ Could not load detailed results")
    results = pd.DataFrame()

# Table 1: Dataset Description
print("\n" + "="*60)
print("📊 TABLE 1: Dataset Description")
print("="*60)

if not df.empty:
    # Calculate statistics
    word_counts_orig = df['review'].str.split().str.len()
    word_counts_clean = df['cleaned_review'].str.split().str.len()
    
    table1 = pd.DataFrame({
        'Feature': [
            'Total Reviews',
            'Positive Reviews',
            'Negative Reviews',
            'Average Words (original)',
            'Average Words (cleaned)',
            'Vocabulary Size (TF-IDF)'
        ],
        'Value': [
            str(len(df)),
            f"{sum(df['sentiment']=='positive')} ({(sum(df['sentiment']=='positive')/len(df)*100):.1f}%)",
            f"{sum(df['sentiment']=='negative')} ({(sum(df['sentiment']=='negative')/len(df)*100):.1f}%)",
            f"{word_counts_orig.mean():.1f}",
            f"{word_counts_clean.mean():.1f}",
            "500"
        ]
    })
    print(table1.to_string(index=False))
else:
    print("No dataset information available")

# Table 2: Confusion Matrices
print("\n" + "="*60)
print("📊 TABLE 2: Confusion Matrices")
print("="*60)

try:
    results_basic = pd.read_csv('output/results.csv')
    for _, row in results_basic.iterrows():
        print(f"\n{row['Model']}:")
        print("              Predicted")
        print("              Neg  Pos")
        print(f"Actual Neg  {int(row['TN']):3d}  {int(row['FP']):3d}")
        print(f"       Pos  {int(row['FN']):3d}  {int(row['TP']):3d}")
except:
    print("No confusion matrix data available")

# Table 3: Performance Metrics
print("\n" + "="*60)
print("📊 TABLE 3: Performance Metrics")
print("="*60)

if not results.empty:
    # Format for display
    results_display = results.copy()
    for col in results_display.columns:
        if col != 'MCC':
            results_display[col] = results_display[col].apply(lambda x: f"{x:.2%}")
        else:
            results_display[col] = results_display[col].apply(lambda x: f"{x:.3f}")
    print(results_display.to_string())
else:
    print("No performance metrics available")

# Table 4: Sample Reviews
print("\n" + "="*60)
print("📊 TABLE 4: Sample Reviews")
print("="*60)

if not df.empty:
    samples = []
    for sentiment in ['positive', 'negative']:
        sample_df = df[df['sentiment'] == sentiment].head(2)
        for _, row in sample_df.iterrows():
            review_short = row['review'][:100] + '...' if len(row['review']) > 100 else row['review']
            cleaned_short = row['cleaned_review'][:50] + '...' if len(row['cleaned_review']) > 50 else row['cleaned_review']
            samples.append({
                'Review': review_short,
                'Sentiment': sentiment.capitalize(),
                'Cleaned': cleaned_short
            })
    
    if samples:
        samples_df = pd.DataFrame(samples)
        print(samples_df.to_string(index=False))
    else:
        print("No sample reviews available")

# Save all tables
try:
    with pd.ExcelWriter('output/report_tables.xlsx') as writer:
        if not df.empty:
            table1.to_excel(writer, sheet_name='Dataset', index=False)
        
        if not results.empty:
            results.to_excel(writer, sheet_name='Metrics')
        
        if 'samples_df' in locals():
            samples_df.to_excel(writer, sheet_name='Samples', index=False)
    
    print(f"\n✅ Report tables saved to output/report_tables.xlsx")
except Exception as e:
    print(f"\n⚠️ Could not save Excel file: {e}")

print("\n" + "="*60)
print("✅ REPORT GENERATION COMPLETE")
print("="*60)
