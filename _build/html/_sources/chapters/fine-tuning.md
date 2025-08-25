# Fine-Tuning AI Models

## Introduction to Fine-Tuning

Fine-tuning is the process of adapting a pre-trained model to perform better on a specific task or domain. Rather than training a model from scratch, fine-tuning leverages the knowledge already learned by large models and specializes them for particular use cases.

## Why Fine-Tune?

### Advantages of Fine-Tuning
1. **Reduced Training Time**: Start with pre-trained weights
2. **Lower Data Requirements**: Effective with smaller datasets
3. **Better Performance**: Often outperforms training from scratch
4. **Cost Effective**: Requires less computational resources
5. **Domain Adaptation**: Customize models for specific industries or use cases

### When to Fine-Tune
- When you have domain-specific data
- When the base model doesn't perform well on your task
- When you need better control over model behavior
- When you want to reduce model size or inference time

## Types of Fine-Tuning

### Full Fine-Tuning
Updating all parameters of the pre-trained model.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FullFineTuningModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(
            self.backbone.config.hidden_size, 
            num_classes
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training loop
def train_full_finetuning(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)
Updates only a small number of additional parameters.

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# Apply LoRA to model
model = AutoModel.from_pretrained("bert-base-uncased")
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 109,514,298 || trainable%: 0.27%
```

#### Adapter Layers
Small bottleneck layers inserted between transformer layers.

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Residual connection
        residual = x
        
        # Adapter computation
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        
        # Add residual
        return x + residual
```

#### Prefix Tuning
Learning soft prompts that guide model behavior.

```python
class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length=10):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        self.prefix_tokens = nn.Parameter(
            torch.randn(prefix_length, model.config.hidden_size)
        )
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        
        # Expand prefix for batch
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get input embeddings
        inputs_embeds = self.model.embeddings(input_ids)
        
        # Concatenate prefix with input embeddings
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        
        # Adjust attention mask
        prefix_mask = torch.ones(batch_size, self.prefix_length).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
```

## Fine-Tuning Large Language Models

### Using Hugging Face Transformers

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import torch

class LLMFineTuner:
    def __init__(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, texts, labels):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        dataset = Dataset.from_dict({'text': texts, 'labels': labels})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        trainer.train()
        
        return trainer
```

### Custom Fine-Tuning Loop

```python
def custom_fine_tuning_loop(model, train_loader, val_loader, num_epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model
```

## Fine-Tuning Strategies

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Reduce LR on plateau
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3,
    verbose=True
)

# Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Initial restart period
    T_mult=2,  # Factor to increase restart period
    eta_min=1e-7  # Minimum learning rate
)
```

### Gradual Unfreezing

```python
def gradual_unfreezing(model, optimizer, train_loader, unfreeze_schedule):
    """
    Gradually unfreeze layers during training
    
    Args:
        model: The model to train
        optimizer: The optimizer
        train_loader: Training data loader
        unfreeze_schedule: Dict mapping epoch to layers to unfreeze
    """
    
    # Initially freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    for epoch in range(num_epochs):
        # Check if we need to unfreeze layers
        if epoch in unfreeze_schedule:
            layers_to_unfreeze = unfreeze_schedule[epoch]
            for layer_name in layers_to_unfreeze:
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Re-initialize optimizer with new parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=2e-5
            )
        
        # Training loop for this epoch
        train_epoch(model, train_loader, optimizer)
```

### Domain Adaptation

```python
class DomainAdaptationTrainer:
    def __init__(self, source_model, target_domain_data):
        self.source_model = source_model
        self.target_data = target_domain_data
    
    def domain_adversarial_training(self):
        """
        Train with domain adversarial loss to learn domain-invariant features
        """
        # Add domain classifier
        domain_classifier = nn.Sequential(
            nn.Linear(self.source_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Source vs Target domain
        )
        
        # Gradient reversal layer for adversarial training
        class GradientReversalLayer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)
            
            @staticmethod
            def backward(ctx, grad_output):
                output = grad_output.neg() * ctx.alpha
                return output, None
        
        # Training loop with domain adversarial loss
        for batch in train_loader:
            # Task-specific loss
            task_loss = compute_task_loss(batch)
            
            # Domain adversarial loss
            features = self.source_model.get_hidden_states(batch)
            reversed_features = GradientReversalLayer.apply(features, alpha=0.1)
            domain_predictions = domain_classifier(reversed_features)
            domain_loss = compute_domain_loss(domain_predictions, batch['domain_labels'])
            
            # Combined loss
            total_loss = task_loss + domain_loss
            total_loss.backward()
```

## Best Practices

### Data Preparation
1. **Quality over Quantity**: High-quality, representative data is crucial
2. **Data Augmentation**: Use techniques like paraphrasing, back-translation
3. **Balanced Datasets**: Ensure good representation of all classes
4. **Validation Strategy**: Use proper train/validation/test splits

### Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 2, 10)
    
    # Train model with suggested hyperparameters
    model = train_model(learning_rate, batch_size, num_epochs)
    
    # Return validation metric to optimize
    return evaluate_model(model)

# Run hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")
```

### Monitoring and Debugging

```python
import wandb
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Log custom metrics
        wandb.log({
            "epoch": state.epoch,
            "learning_rate": state.log_history[-1]["learning_rate"],
            "gradient_norm": get_gradient_norm(model)
        })
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save final model
        wandb.save("final_model.pth")

def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)
```

### Model Evaluation and Deployment

```python
def evaluate_fine_tuned_model(model, test_loader, tokenizer):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(classification_report(true_labels, predictions))
    print(confusion_matrix(true_labels, predictions))
    
    return predictions, true_labels

# Model serving
class FineTunedModelServer:
    def __init__(self, model_path, tokenizer_path):
        self.model = torch.load(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        return {
            'predicted_class': predicted_class.item(),
            'probabilities': probabilities.squeeze().tolist()
        }
```

## Common Challenges and Solutions

### Overfitting
- **Problem**: Model performs well on training data but poorly on validation
- **Solutions**: Regularization, dropout, early stopping, data augmentation

### Catastrophic Forgetting
- **Problem**: Fine-tuned model loses original capabilities
- **Solutions**: Multi-task learning, elastic weight consolidation, replay methods

### Limited Data
- **Problem**: Insufficient training data for effective fine-tuning
- **Solutions**: Data augmentation, transfer learning, few-shot learning techniques

### Computational Resources
- **Problem**: Large models require significant computing power
- **Solutions**: Parameter-efficient methods, gradient checkpointing, mixed precision training

---

**Next Chapter**: Learn how to properly evaluate AI model performance using various metrics and methodologies.