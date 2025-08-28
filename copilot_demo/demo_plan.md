# GitHub Copilot Demonstration: Higher-Order Function Coding Task

## Overview

This activity demonstrates how to use GitHub Copilot and LLMs for Python development, including prompt engineering, code generation, test creation, and documentation.

### 1. Coding Task

In Python, write a function that returns the product of the first and last odd numbers in the given list. If there's only one odd number, square it. If no odd numbers exist, return None.

#### Steps

1. Introduce ask, edit, and agent modes of GitHub Copilot.
2. Switch between different models (Claude Sonnet 4 is the best I used so far).
3. Demo how to add different files you want to work with.
4. `Command + I` to do inline code generation (works in both file + terminal)
5. `Command + Ctrl + I` to summon chat box.
6. Highlight code and `Command + Shift + I` to do ask about the highlighted code section only.
7. Create a personal prompt to instruct LLM not to write code for you, but to explain the concepts and guide you to write the code yourself. Example prompt:

   - **"I want you to act as a coding tutor. I will provide you with a coding problem, and you will guide me through the process of solving it step by step. Instead of writing the code for me, please explain the concepts involved and ask me questions to help me arrive at the solution on my own."**

Show how to call the prompt using `/`

8. Prompt about the coding task, **"which programming concepts are used in this question. What are they?"**
9. Ask ChatGPT to "give an example of using each concept in a new context (unrelated to the problem above). Explain to me like I'm 10 years old."
10. Break down the problem into smaller steps and ask LLM to help you with each step.
    - Example prompts:
      - **"How do I find the first odd number in a list?"**
      - **"How do I find the last odd number in a list?"**
      - **"How do I multiply two numbers in Python?"**
      - **"How do I check if a list has no odd numbers?"**
      - **"How do I check if a list has only one odd number?"**
11. **Test Generation**: Use LLM to write tests and create toy/test data.
12. Implement the function step by step with LLM assistance.
13. **Docstring generation**: LLM could be used for documentation generation.
14. What edge cases are not considered in the `odd_function`? How to improve it?

### 2. Debugging Task

1. In the terminal, run `claude` and see the bug show up in terminal. 
2. Copy the error message and paste it into the chat box. Ask LLM that **"I'm pretty sure I have installed claude code yesterday, I remember also installing node.js, nvm etc. But why can't I find it anymore?"**
3. Show how the answer LLM give is wrong. --> Shows how it is important for you to know what you are doing.
4. Ask LLM to **"explain the concepts of nvm, npm, node.js like I'm 5 years old. Please use analogies of animals."** --> Shows how LLM tailor explanation to yoru own interest.
