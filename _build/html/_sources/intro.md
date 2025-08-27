# AI Tools & Prompt Engineering for Data Science

Welcome to your Master of Data Science program! This guide introduces essential AI tools and prompt engineering techniques that will enhance your data science workflow.

## Essential AI Tools for Data Scientists

### 💬 Large Language Models (LLMs)
**Primary Tools:**
- **ChatGPT (OpenAI)**: Versatile for code generation, debugging, and explanations
- **Claude (Anthropic)**: Excellent for analytical thinking and code review
- **Gemini (Google)**: Strong integration with Google ecosystem

**Use Cases:**
- Code generation and debugging
- Data analysis explanations
- Documentation writing
- Research assistance

### 🔍 AI-Powered Research & Analysis
- **Perplexity**: AI-powered search engine with source citations
- **NotebookLM (Google)**: Document analysis and study guide generation
- **Elicit**: Scientific literature review and synthesis

### 👨‍💻 AI-Enhanced Development Environments
- **Cursor IDE**: AI-powered code editor with context awareness
- **GitHub Copilot**: AI pair programmer integrated into VS Code
- **Replit**: Browser-based coding with AI assistance

### 📊 Data Science Specific Tools
- **Julius AI**: Natural language data analysis
- **DataCamp Workspace**: AI-assisted data science projects
- **Kaggle Notebooks**: Integrated AI code suggestions

### 🎨 Content Creation & Visualization
- **Canva AI**: Data visualization and presentation design
- **Gamma**: AI-powered presentation creation
- **Tome**: Interactive presentation builder

### 🧪 Model Comparison & Evaluation
- **LMArena**: Compare different language models side-by-side
- **OpenAI Playground**: Experiment with model parameters
- **Hugging Face Spaces**: Test and compare open-source models

## Prompt Engineering Fundamentals

Prompt engineering is the practice of designing effective inputs to get optimal outputs from AI models. For data scientists, this skill is crucial for automating workflows and enhancing productivity.

### Core Principles

#### 1. Be Clear and Specific
```{admonition} Best Practice
:class: tip

Vague prompts lead to unpredictable outputs. Specify exactly what you need.
```

**Poor prompt**: "Help with my data analysis"
**Better prompt**: "Review this Python pandas code for analyzing customer churn. Identify potential bugs and suggest performance improvements."

#### 2. Provide Context and Role
Give the AI context about your specific situation:

```
Context: You are a senior data scientist reviewing a machine learning pipeline.
Task: Explain the difference between precision and recall metrics.
Audience: New data science master's students
Format: Brief explanation with a practical example
```

### Essential Techniques

#### 1. Zero-Shot vs Few-Shot Prompting

**Zero-Shot**: Ask without examples
```
Classify this customer review sentiment as positive, negative, or neutral:
"The product works as expected but delivery was slow."
```

**Few-Shot**: Provide examples to guide behavior
```
Classify customer feedback sentiment:

Example 1: "Amazing product, fast shipping!" → Positive
Example 2: "Product broke after one week." → Negative
Example 3: "It's okay, nothing special." → Neutral

Now classify: "Great quality but expensive." → ?
```

#### 2. Chain-of-Thought (CoT) Prompting
Encourage step-by-step reasoning for complex problems:

```
Analyze this A/B test result step by step:
Control group: 1000 users, 50 conversions
Test group: 1000 users, 65 conversions

Please:
1. Calculate conversion rates for both groups
2. Determine if the difference is statistically significant
3. Provide a recommendation based on the results
```

#### 3. Role-Based Prompting
Assign specific expertise to the AI:

```
You are an experienced MLOps engineer. Review this model deployment code and suggest improvements for:
- Scalability
- Monitoring
- Error handling
- Security best practices
```

### Advanced Strategies

#### 1. Structured Output Format
Request specific formats for consistent results:

```
Analyze this dataset and respond in JSON format:
{
  "data_quality": "assessment here",
  "missing_values": "percentage and pattern",
  "recommendations": ["action 1", "action 2", "action 3"],
  "next_steps": "prioritized list"
}
```

#### 2. Iterative Refinement
Use follow-up prompts to improve responses:

```
Initial: "Explain logistic regression"
Follow-up: "Now provide the mathematical formula"
Refinement: "Add a Python implementation example"
Final: "Include when to use vs. when to avoid this algorithm"
```

#### 3. Self-Consistency and Validation
Ask the AI to verify its own work:

```
Write a function to calculate correlation coefficient.

Now review your code above:
1. Check for any bugs or edge cases
2. Suggest improvements for efficiency
3. Add proper error handling
```

### Practical Applications for Data Scientists

#### Code Generation and Review
```
Task: Create a Python function to preprocess text data
Requirements:
- Remove special characters
- Convert to lowercase
- Handle missing values
- Return cleaned text and metadata about changes made
- Include type hints and docstring
```

#### Data Analysis Explanations
```
Explain this correlation matrix interpretation to stakeholders:
[Include your correlation matrix]

Requirements:
- Non-technical language
- Focus on business implications
- Highlight key relationships
- Suggest actionable insights
```

#### Model Documentation
```
Generate documentation for this machine learning model:

Model type: Random Forest Classifier
Purpose: Customer churn prediction
Features: [list your features]
Performance: 85% accuracy, 0.82 F1-score

Include:
- Model description
- Feature importance interpretation
- Limitations and assumptions
- Deployment considerations
```

### Best Practices for Data Science Workflows

#### 1. Version Control Your Prompts
- Document successful prompts in your project repository
- Track what works for different types of tasks
- Share effective prompts with your team

#### 2. Validate AI-Generated Code
```{admonition} Critical Practice
:class: warning

Always test AI-generated code with sample data before using in production.
```

**Testing Strategy:**
1. Create small test datasets
2. Verify outputs manually for a subset
3. Check edge cases and error handling
4. Compare with alternative implementations

#### 3. Combine Multiple AI Tools
- Use ChatGPT for initial code generation
- Use Claude for code review and optimization
- Use Perplexity for research and documentation
- Compare outputs using LMArena when unsure

#### 4. Ethical Considerations
- Always cite AI assistance in academic work
- Verify factual claims from AI responses
- Be transparent about AI usage in professional settings
- Understand your organization's AI usage policies

### Quick Reference: Prompt Templates

#### Data Analysis Request
```
Context: Data scientist analyzing [dataset type]
Task: [specific analysis needed]
Data: [brief description of data structure]
Output format: [specify format - code, report, visualization, etc.]
Constraints: [any limitations or requirements]
```

#### Code Review Template
```
Please review this [language] code for [purpose]:

[code here]

Focus on:
- Correctness and logic
- Performance optimization
- Best practices adherence
- Security considerations
- Readability and maintainability
```

#### Learning and Explanation Template
```
Explain [concept] for a data science context:
- Target audience: [specify level]
- Include practical example with [specific domain]
- Compare with [alternative approaches]
- When to use vs. when to avoid
```

---

## Getting Started

1. **Choose your primary AI tool**: Start with ChatGPT or Claude for general data science tasks
2. **Practice with small tasks**: Begin with code review and simple analysis questions
3. **Build a prompt library**: Save effective prompts for common data science workflows
4. **Experiment with different approaches**: Compare zero-shot vs few-shot for your use cases
5. **Join the community**: Use LMArena to understand model strengths and weaknesses

**Remember**: AI tools are powerful assistants, but critical thinking and domain expertise remain essential. Use AI to enhance your capabilities, not replace your analytical skills.

---

*Welcome to your data science journey! These tools and techniques will help you work more efficiently and explore new possibilities in your projects.*