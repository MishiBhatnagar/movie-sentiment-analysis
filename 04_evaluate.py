# 04_evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import pickle
import os

print("="*60)
print("📈 STEP 4: DETAILED EVALUATION")
print("="*60)

# Create figures folder
os.makedirs('output/figures', exist_ok=True)

class DetailedEvaluator:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        # Load features and labels
        X = np.load('output/tfidf_features.npy')
        y = np.load('output/labels.npy')
        
        # Split data (same as training)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Load models
        model_files = ['naive_bayes.pkl', 'svm.pkl', 'logistic_regression.pkl']
        model_names = ['Naive Bayes', 'SVM', 'Logistic Regression']
        
        for name, file in zip(model_names, model_files):
            try:
                with open(f'output/{file}', 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"✅ Loaded {name}")
            except:
                print(f"⚠️ Could not load {file}")
        
        print(f"\n📂 Test data: {X_test.shape[0]} samples")
        
        return X_test, y_test
    
    def calculate_all_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        metrics = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'FDR': fp / (fp + tp) if (fp + tp) > 0 else 0,
        }
        
        # MCC
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['MCC'] = numerator / denominator if denominator > 0 else 0
        
        return metrics, cm
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics, cm = self.calculate_all_metrics(y_test, y_pred)
            self.metrics[name] = metrics
            
            print(f"\n📊 {name}:")
            print(f"   Accuracy:    {metrics['Accuracy']:.2%}")
            print(f"   Precision:   {metrics['Precision']:.2%}")
            print(f"   Recall:      {metrics['Recall']:.2%}")
            print(f"   F1-Score:    {metrics['F1-Score']:.2%}")
            print(f"   Specificity: {metrics['Specificity']:.2%}")
            print(f"   MCC:         {metrics['MCC']:.3f}")
            
            if cm.shape == (2,2):
                print(f"\n   Confusion Matrix:")
                print(f"               Predicted")
                print(f"               Neg  Pos")
                print(f"   Actual Neg  {cm[0,0]:3d}  {cm[0,1]:3d}")
                print(f"          Pos  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    def create_comparison_table(self):
        """Create comparison table"""
        if not self.metrics:
            return None
            
        df = pd.DataFrame(self.metrics).T
        
        # Create a formatted version
        df_display = df.copy()
        for col in df_display.columns:
            if col != 'MCC':
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
        
        print("\n" + "="*70)
        print("📊 COMPLETE RESULTS TABLE")
        print("="*70)
        print(df_display.to_string())
        
        df.to_csv('output/detailed_results.csv')
        print(f"\n✅ Saved to output/detailed_results.csv")
        
        return df
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices"""
        n_models = len(self.models)
        if n_models == 0:
            return
            
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/figures/confusion_matrices.png', dpi=300)
        plt.show()
        print("✅ Saved confusion_matrices.png")
    
    def plot_metrics_comparison(self):
        """Plot metrics comparison"""
        if not self.metrics:
            return
            
        models = list(self.metrics.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            values = [self.metrics[model][m] for m in metrics]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=model)
            
            for bar, val, metric in zip(bars, values, metrics):
                if metric == 'MCC':
                    label = f'{val:.3f}'
                else:
                    label = f'{val:.1%}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       label, ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('output/figures/metrics_comparison.png', dpi=300)
        plt.show()
        print("✅ Saved metrics_comparison.png")

# Run evaluation
if __name__ == "__main__":
    evaluator = DetailedEvaluator()
    X_test, y_test = evaluator.load_models_and_data()
    
    if evaluator.models:
        evaluator.evaluate_all(X_test, y_test)
        evaluator.create_comparison_table()
        evaluator.plot_confusion_matrices(X_test, y_test)
        evaluator.plot_metrics_comparison()
        print("\n✅ EVALUATION COMPLETE!")
    else:
        print("\n❌ No models loaded. Run training first!")
