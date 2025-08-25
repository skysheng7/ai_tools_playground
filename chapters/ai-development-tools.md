# AI Development Tools and Frameworks

## Overview of the AI Development Ecosystem

The AI development landscape offers a rich ecosystem of tools, frameworks, and platforms designed to streamline the development, deployment, and management of AI applications.

## Machine Learning Frameworks

### TensorFlow
Google's open-source machine learning framework.

**Key Features:**
- Comprehensive ecosystem (TensorFlow Lite, TensorFlow.js, TensorFlow Serving)
- Strong production deployment capabilities
- Excellent documentation and community support

```python
import tensorflow as tf

# Simple neural network example
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### PyTorch
Facebook's dynamic deep learning framework.

**Key Features:**
- Dynamic computation graphs
- Pythonic and intuitive API
- Strong research community adoption

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Hugging Face Transformers
The go-to library for working with pre-trained language models.

```python
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Tokenize and encode text
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

## Development Environments

### Jupyter Notebooks
Interactive development environment for data science and AI experimentation.

**Benefits:**
- Interactive code execution
- Rich media output (plots, images, tables)
- Documentation alongside code
- Easy sharing and collaboration

### Google Colab
Cloud-based Jupyter environment with free GPU access.

**Features:**
- Free access to GPUs and TPUs
- Pre-installed ML libraries
- Easy sharing and collaboration
- Integration with Google Drive

### Kaggle Notebooks
Competitive data science platform with hosted notebooks.

### Local Development
Setting up local AI development environments:

```bash
# Create virtual environment
python -m venv ai_env
source ai_env/bin/activate  # On Windows: ai_env\Scripts\activate

# Install common packages
pip install torch torchvision transformers
pip install tensorflow scikit-learn pandas numpy matplotlib
pip install jupyter notebook
```

## Cloud AI Platforms

### Amazon Web Services (AWS)
- **SageMaker**: Fully managed ML platform
- **Bedrock**: Managed foundation models
- **Comprehend**: Natural language processing
- **Rekognition**: Computer vision services

### Google Cloud Platform (GCP)
- **Vertex AI**: Unified ML platform
- **AutoML**: Automated model training
- **Natural Language AI**: Pre-trained NLP models
- **Vision AI**: Image analysis services

### Microsoft Azure
- **Azure Machine Learning**: End-to-end ML lifecycle
- **Cognitive Services**: Pre-built AI capabilities
- **OpenAI Service**: Access to GPT models
- **Form Recognizer**: Document processing

### OpenAI API
Direct access to state-of-the-art language models:

```python
import openai

# Set up API key
openai.api_key = "your-api-key"

# Generate text
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

## Model Development Tools

### MLflow
Open-source platform for the complete ML lifecycle.

**Features:**
- Experiment tracking
- Model packaging and deployment
- Model registry
- Model monitoring

```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = RandomForestClassifier(max_depth=3, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Weights & Biases (wandb)
Experiment tracking and model management platform.

```python
import wandb

# Initialize wandb
wandb.init(project="my-ai-project")

# Log metrics during training
for epoch in range(num_epochs):
    loss = train_epoch(model, dataloader)
    wandb.log({"epoch": epoch, "loss": loss})
```

### DVC (Data Version Control)
Version control for machine learning projects.

```bash
# Initialize DVC
dvc init

# Add data to DVC tracking
dvc add data/training_data.csv

# Create pipeline
dvc run -d data/training_data.csv \
        -o models/model.pkl \
        python train.py
```

## Specialized AI Tools

### LangChain
Framework for developing applications with language models.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a blog post about {topic}"
)

# Create chain
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# Generate content
result = chain.run("artificial intelligence")
```

### Gradio
Create web interfaces for AI models quickly.

```python
import gradio as gr

def classify_image(image):
    # Your image classification logic here
    return "cat"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

interface.launch()
```

### Streamlit
Build interactive web apps for AI/ML projects.

```python
import streamlit as st
import pandas as pd

st.title("AI Model Dashboard")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    
    if st.button("Run Prediction"):
        # Your prediction logic here
        st.write("Predictions completed!")
```

## Best Practices

### Environment Management
- Use virtual environments or containers
- Pin dependency versions
- Document environment setup
- Use requirements.txt or environment.yml files

### Code Organization
- Separate data processing, model training, and evaluation
- Use configuration files for hyperparameters
- Implement proper logging
- Write unit tests for critical functions

### Experiment Tracking
- Track all experiments systematically
- Log hyperparameters, metrics, and artifacts
- Use version control for code and data
- Document insights and learnings

### Reproducibility
- Set random seeds
- Use deterministic algorithms when possible
- Container deployment (Docker)
- Environment snapshots

---

**Next Chapter**: Explore real-world applications and practical implementations of AI tools.