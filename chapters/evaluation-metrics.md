# Evaluation Metrics for AI Models

## Introduction to Model Evaluation

Proper evaluation is crucial for understanding model performance, comparing different approaches, and ensuring models meet business requirements. This chapter covers comprehensive evaluation strategies for various types of AI models.

## Classification Metrics

### Basic Metrics

#### Accuracy
The proportion of correct predictions among total predictions.

```python
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Example
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")  # Output: 0.333
```

#### Precision, Recall, and F1-Score

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }

# Detailed classification report
def detailed_classification_report(y_true, y_pred, class_names=None):
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Convert to DataFrame for better visualization
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    return df

# Example usage
class_names = ['Class A', 'Class B', 'Class C']
report_df = detailed_classification_report(y_true, y_pred, class_names)
print(report_df)
```

### Advanced Classification Metrics

#### ROC-AUC and PR-AUC

```python
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

class ClassificationEvaluator:
    def __init__(self, y_true, y_pred_proba, y_pred):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
    
    def roc_auc_analysis(self, plot=True):
        """Calculate ROC-AUC for binary/multiclass classification"""
        if len(np.unique(self.y_true)) == 2:
            # Binary classification
            auc_score = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
            
            if plot:
                fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.show()
        else:
            # Multiclass classification
            auc_score = roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr')
        
        return auc_score
    
    def precision_recall_analysis(self, plot=True):
        """Calculate PR-AUC"""
        if len(np.unique(self.y_true)) == 2:
            # Binary classification
            pr_auc = average_precision_score(self.y_true, self.y_pred_proba[:, 1])
            
            if plot:
                precision, recall, _ = precision_recall_curve(
                    self.y_true, self.y_pred_proba[:, 1]
                )
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.show()
        else:
            # Multiclass - calculate average
            pr_auc = average_precision_score(self.y_true, self.y_pred_proba, average='weighted')
        
        return pr_auc
    
    def confusion_matrix_analysis(self, normalize=None):
        """Generate and visualize confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred, normalize=normalize)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = np.unique(self.y_true)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        
        return cm
```

## Regression Metrics

### Standard Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    
    def calculate_all_metrics(self):
        """Calculate comprehensive regression metrics"""
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(self.y_true, self.y_pred)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(self.y_true, self.y_pred)
        
        # R-squared
        metrics['r2'] = r2_score(self.y_true, self.y_pred)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = self.mean_absolute_percentage_error()
        
        # Median Absolute Error
        metrics['median_ae'] = np.median(np.abs(self.y_true - self.y_pred))
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(self.y_true - self.y_pred))
        
        return metrics
    
    def mean_absolute_percentage_error(self):
        """Calculate MAPE, handling zero values"""
        mask = self.y_true != 0
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def residual_analysis(self, plot=True):
        """Analyze residuals for model diagnostics"""
        residuals = self.y_true - self.y_pred
        
        analysis = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': self.calculate_skewness(residuals),
            'residual_kurtosis': self.calculate_kurtosis(residuals)
        }
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Residuals vs Predicted
            axes[0, 0].scatter(self.y_pred, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            
            # Q-Q plot for normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            
            # Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals')
            
            # Actual vs Predicted
            axes[1, 1].scatter(self.y_true, self.y_pred, alpha=0.6)
            min_val = min(self.y_true.min(), self.y_pred.min())
            max_val = max(self.y_true.max(), self.y_pred.max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Actual vs Predicted')
            
            plt.tight_layout()
            plt.show()
        
        return analysis
    
    def calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis
```

## Natural Language Processing Metrics

### Text Generation Metrics

```python
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer

class TextGenerationEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
    
    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score for text generation"""
        # Tokenize sentences
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        
        # Calculate BLEU scores
        bleu_scores = {}
        bleu_scores['bleu1'] = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0))
        bleu_scores['bleu2'] = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_scores['bleu3'] = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu_scores['bleu4'] = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        return bleu_scores
    
    def calculate_rouge_score(self, reference, candidate):
        """Calculate ROUGE scores for text summarization"""
        scores = self.rouge_scorer.score(reference, candidate)
        
        rouge_scores = {}
        for key, value in scores.items():
            rouge_scores[f'{key}_precision'] = value.precision
            rouge_scores[f'{key}_recall'] = value.recall
            rouge_scores[f'{key}_fmeasure'] = value.fmeasure
        
        return rouge_scores
    
    def calculate_perplexity(self, model, text, tokenizer):
        """Calculate perplexity of generated text"""
        import torch
        
        # Tokenize text
        tokens = tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def calculate_diversity_metrics(self, generated_texts):
        """Calculate diversity metrics for generated text"""
        # Distinct-n metrics
        def distinct_n(texts, n):
            all_ngrams = []
            total_ngrams = 0
            
            for text in texts:
                tokens = text.split()
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.extend(ngrams)
                total_ngrams += len(ngrams)
            
            unique_ngrams = len(set(all_ngrams))
            return unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        diversity_metrics = {
            'distinct_1': distinct_n(generated_texts, 1),
            'distinct_2': distinct_n(generated_texts, 2),
            'distinct_3': distinct_n(generated_texts, 3),
            'vocab_size': len(set(' '.join(generated_texts).split()))
        }
        
        return diversity_metrics
```

### Semantic Similarity Metrics

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def batch_semantic_similarity(self, texts1, texts2):
        """Calculate semantic similarity for batches of texts"""
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(sim)
        
        return similarities
    
    def evaluate_retrieval(self, queries, documents, relevant_docs):
        """Evaluate information retrieval performance"""
        query_embeddings = self.model.encode(queries)
        doc_embeddings = self.model.encode(documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        # Calculate metrics for each query
        metrics = {'precision_at_k': [], 'recall_at_k': [], 'map': []}
        
        for i, (query_sim, relevant) in enumerate(zip(similarities, relevant_docs)):
            # Get top-k documents
            top_k_indices = np.argsort(query_sim)[::-1][:10]
            
            # Calculate precision@k and recall@k
            relevant_retrieved = sum(1 for idx in top_k_indices if idx in relevant)
            precision_k = relevant_retrieved / len(top_k_indices)
            recall_k = relevant_retrieved / len(relevant) if relevant else 0
            
            metrics['precision_at_k'].append(precision_k)
            metrics['recall_at_k'].append(recall_k)
        
        # Average metrics
        avg_metrics = {
            'avg_precision_at_10': np.mean(metrics['precision_at_k']),
            'avg_recall_at_10': np.mean(metrics['recall_at_k'])
        }
        
        return avg_metrics
```

## Computer Vision Metrics

### Image Classification Metrics

```python
import cv2
import numpy as np
from sklearn.metrics import top_k_accuracy_score

class VisionEvaluator:
    def __init__(self):
        pass
    
    def top_k_accuracy(self, y_true, y_pred_proba, k=5):
        """Calculate top-k accuracy"""
        return top_k_accuracy_score(y_true, y_pred_proba, k=k)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map(self, pred_boxes, pred_scores, pred_labels, 
                     true_boxes, true_labels, iou_threshold=0.5):
        """Calculate mean Average Precision for object detection"""
        # This is a simplified version - real implementation would be more complex
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        true_positives = []
        false_positives = []
        
        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx]
            
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(true_boxes, true_labels)):
                if pred_label == gt_label:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recalls = tp_cumsum / len(true_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate Average Precision
        ap = np.trapz(precisions, recalls)
        
        return ap
    
    def pixel_accuracy(self, y_true, y_pred):
        """Calculate pixel accuracy for semantic segmentation"""
        return np.mean(y_true == y_pred)
    
    def dice_coefficient(self, y_true, y_pred):
        """Calculate Dice coefficient for segmentation"""
        intersection = np.sum(y_true * y_pred)
        return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
```

## Model Fairness and Bias Evaluation

### Fairness Metrics

```python
class FairnessEvaluator:
    def __init__(self, y_true, y_pred, sensitive_attribute):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_attribute = sensitive_attribute
    
    def demographic_parity(self):
        """Calculate demographic parity difference"""
        groups = np.unique(self.sensitive_attribute)
        acceptance_rates = {}
        
        for group in groups:
            group_mask = self.sensitive_attribute == group
            acceptance_rate = np.mean(self.y_pred[group_mask])
            acceptance_rates[group] = acceptance_rate
        
        # Calculate maximum difference
        rates = list(acceptance_rates.values())
        parity_difference = max(rates) - min(rates)
        
        return parity_difference, acceptance_rates
    
    def equalized_odds(self):
        """Calculate equalized odds difference"""
        groups = np.unique(self.sensitive_attribute)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in groups:
            group_mask = self.sensitive_attribute == group
            group_y_true = self.y_true[group_mask]
            group_y_pred = self.y_pred[group_mask]
            
            # True Positive Rate
            tpr = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true == 1)
            tpr_by_group[group] = tpr
            
            # False Positive Rate
            fpr = np.sum((group_y_true == 0) & (group_y_pred == 1)) / np.sum(group_y_true == 0)
            fpr_by_group[group] = fpr
        
        # Calculate differences
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_difference = max(tpr_values) - min(tpr_values)
        fpr_difference = max(fpr_values) - min(fpr_values)
        
        return {
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group
        }
    
    def calibration_analysis(self, y_pred_proba):
        """Analyze prediction calibration across groups"""
        from sklearn.calibration import calibration_curve
        
        groups = np.unique(self.sensitive_attribute)
        calibration_results = {}
        
        for group in groups:
            group_mask = self.sensitive_attribute == group
            group_y_true = self.y_true[group_mask]
            group_y_pred_proba = y_pred_proba[group_mask]
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                group_y_true, group_y_pred_proba, n_bins=10
            )
            
            calibration_results[group] = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        
        return calibration_results
```

## Cross-Validation and Model Selection

### Advanced Cross-Validation Strategies

```python
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate
)

class ModelValidator:
    def __init__(self, model, X, y, groups=None):
        self.model = model
        self.X = X
        self.y = y
        self.groups = groups
    
    def stratified_kfold_validation(self, k=5, scoring='accuracy'):
        """Perform stratified k-fold cross-validation"""
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            self.model, self.X, self.y, 
            cv=skf, scoring=scoring
        )
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'confidence_interval': self.calculate_confidence_interval(scores)
        }
    
    def time_series_validation(self, n_splits=5, scoring='neg_mean_squared_error'):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = cross_val_score(
            self.model, self.X, self.y,
            cv=tscv, scoring=scoring
        )
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    def group_kfold_validation(self, k=5, scoring='accuracy'):
        """Perform group k-fold cross-validation"""
        if self.groups is None:
            raise ValueError("Groups must be provided for group k-fold validation")
        
        gkf = GroupKFold(n_splits=k)
        
        scores = cross_val_score(
            self.model, self.X, self.y,
            groups=self.groups, cv=gkf, scoring=scoring
        )
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    def comprehensive_evaluation(self, scoring_metrics):
        """Perform comprehensive model evaluation with multiple metrics"""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = cross_validate(
            self.model, self.X, self.y,
            cv=skf, scoring=scoring_metrics,
            return_train_score=True
        )
        
        results = {}
        for metric in scoring_metrics:
            results[metric] = {
                'test_scores': cv_results[f'test_{metric}'],
                'train_scores': cv_results[f'train_{metric}'],
                'test_mean': np.mean(cv_results[f'test_{metric}']),
                'test_std': np.std(cv_results[f'test_{metric}']),
                'train_mean': np.mean(cv_results[f'train_{metric}']),
                'train_std': np.std(cv_results[f'train_{metric}'])
            }
        
        return results
    
    def calculate_confidence_interval(self, scores, confidence=0.95):
        """Calculate confidence interval for cross-validation scores"""
        from scipy import stats
        
        n = len(scores)
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            confidence, n-1, loc=mean_score, scale=std_error
        )
        
        return ci
```

## Automated Model Evaluation Pipeline

```python
class AutomatedEvaluationPipeline:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, y_pred=None, y_pred_proba=None):
        """Comprehensive automated evaluation"""
        
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        if self.task_type == 'classification':
            return self._evaluate_classification(y_test, y_pred, y_pred_proba)
        elif self.task_type == 'regression':
            return self._evaluate_regression(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive classification evaluation"""
        evaluator = ClassificationEvaluator(y_true, y_pred_proba, y_pred)
        
        results = {
            'basic_metrics': calculate_classification_metrics(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            results['roc_auc'] = evaluator.roc_auc_analysis(plot=False)
            results['pr_auc'] = evaluator.precision_recall_analysis(plot=False)
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred):
        """Comprehensive regression evaluation"""
        evaluator = RegressionEvaluator(y_true, y_pred)
        
        results = {
            'metrics': evaluator.calculate_all_metrics(),
            'residual_analysis': evaluator.residual_analysis(plot=False)
        }
        
        return results
    
    def generate_report(self, results, output_file='evaluation_report.html'):
        """Generate HTML evaluation report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { margin: 10px 0; }
                .section { margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            {content}
        </body>
        </html>
        """
        
        # Generate content based on results
        content = self._generate_report_content(results)
        
        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(html_template.format(content=content))
    
    def _generate_report_content(self, results):
        """Generate HTML content for evaluation results"""
        content = ""
        
        for section, data in results.items():
            content += f"<div class='section'><h2>{section.replace('_', ' ').title()}</h2>"
            
            if isinstance(data, dict):
                content += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for key, value in data.items():
                    content += f"<tr><td>{key}</td><td>{value:.4f if isinstance(value, float) else value}</td></tr>"
                content += "</table>"
            
            content += "</div>"
        
        return content
```

---

**Next Chapter**: Explore the critical ethical considerations when developing and deploying AI systems.