# 00_check_data.py
import pandas as pd
import os

print("="*60)
print("📊 DATASET INFORMATION")
print("="*60)

# Find the data file
data_dir = 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.xlsx')]
if not data_files:
    print("❌ No data file found in data/ folder!")
    exit(1)

filepath = os.path.join(data_dir, data_files[0])
print(f"📂 Found file: {data_files[0]}")

# Load based on file extension
if data_files[0].endswith('.csv'):
    df = pd.read_csv(filepath)
else:
    df = pd.read_excel(filepath)

print(f"\n📊 Dataset Overview:")
print(f"   Total reviews: {len(df)}")
print(f"   Columns: {df.columns.tolist()}")

print(f"\n📊 Sentiment distribution:")
print(df['sentiment'].value_counts())

# Calculate review lengths
df['review_length'] = df['review'].apply(len)
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

print(f"\n📏 Review Statistics:")
print(f"   Average words: {df['word_count'].mean():.1f}")
print(f"   Shortest: {df['word_count'].min()} words")
print(f"   Longest: {df['word_count'].max()} words")

# Show first 3 reviews
print("\n📝 First 3 reviews:")
for i in range(min(3, len(df))):
    print(f"\n{i+1}. [{df['sentiment'].iloc[i]}]")
    print(f"   {df['review'].iloc[i][:150]}...")

# Save info
with open('output/dataset_info.txt', 'w') as f:
    f.write(f"File: {data_files[0]}\n")
    f.write(f"Total reviews: {len(df)}\n")
    f.write(f"Positive: {sum(df['sentiment']=='positive')}\n")
    f.write(f"Negative: {sum(df['sentiment']=='negative')}\n")
    f.write(f"Average words: {df['word_count'].mean():.1f}\n")

print(f"\n✅ Dataset info saved to output/dataset_info.txt")
