# 🧙‍♀️ Chapter 3: Other Large Language Models (LLM) 

*Note: All GenAI tools & models demonstrated below are NOT recommended by UBC due to privacy and safety reasons (but you are NOT restricted from using them)*

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbnV0YzFleHFjNjMyemxzc2M5emkwZTl6cmMyeTl5NW1sYWFsZnZzMSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/50WoQ8JdjyqzhxpBjP/giphy.gif" alt="surprise" style="width: 60%; height: auto;">


## 1. Common large language models (LLM) & interface

### 1.1 Anthropic Claude


```{image} ../images/claude.png
:alt: claude
:width: 100%
```

💡 **Disclaimer**: I have no connection to Anthropic (the company that makes Claude). I recommend using Claude because many programmers and data scientists from the technology industry agree it's the best AI tools for programming, writing and explaining things clearly (as of the time when this tutorial is created)! Personally this is my go-to model that drives everyday tasks.

**Website**: [https://claude.ai](https://claude.ai)

- ✅ Free tier is often enough (Sonnet 4)
- 📧 Requires login
- 🏆 Best reputation among programmers
- 🎨 Claude Artifact feature
- ⛑️ Emphasis on AI safety
- 👐 Transparent with [system prompt](https://docs.anthropic.com/en/release-notes/system-prompts)

*I'm a strong advocate for more transparency in sharing system promopts used to govern the behaviours of generative AI models. System prompts could encode biases and eliminate certain perspectives in the output of generative AI. Anthropic is one of very few companies that publicly shares their system prompts.*

#### Claude artifact demo

Here is an [otter game](https://claude.ai/public/artifacts/5e040757-5e52-4b89-9b8a-dbedb90e07e3) that I created just for fun, using the free tier of Claude and Opus 4.1 model.

You can see the process in this [video](https://youtu.be/CQr79gKsy64).

Prompt I used:

> "You are an expert in game development. Can you help me create a video game using claude artifact? Here is the idea: Otter Breakout/Arkanoid. Otter bounces on its back, juggling a ball (like they do with rocks!). Break ice blocks to free trapped fish. Paddle is a floating otter, ball bounces realistically."

### 1.2 🔒 Other Proprietary Models 

- **Google Gemini 2.5 Pro** - [https://gemini.google.com](https://gemini.google.com)
- **xAI Grok 4** - [https://grok.x.ai](https://grok.x.ai)
- **Mistral Large 2** - [https://mistral.ai](https://mistral.ai)
- And more...

### 1.3 🔑 Open Source/Open Weight Models 

- **Meta Llama 4** - [https://llama.meta.com](https://llama.meta.com)
- **DeepSeek V3.1 MODEL** - [https://deepseek.com](https://deepseek.com) *(Note: Model only, not applications)*
- **Kimi K2** - [https://kimi.moonshot.cn](https://kimi.moonshot.cn)
- And more...

```{image} ../images/models.png
:alt: models
:width: 100%
```

---

## 2. What if I want to try them all?

<img src="https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3cnVnOGQ4bXFwcjdmMGhjeGIzcjI3ZGwyano2bTN6NzZnYW4zdGV0ciZlcD12MV9naWZzX3NlYXJjaCZjdD1n/GiNyo8KD5j9mM/giphy.gif" alt="minion_yeah" style="width: 100%; height: auto;">

### 2.1 Poe (paid)

```{image} ../images/poe.png
:alt: poe
:width: 100%
```

- Accessible via both online chatbot interface & PC/laptop software
- **Online chatbot interface**: [https://poe.com](https://poe.com)
- Access both open source + proprietary models
- **Subscription based** (Note: $20 limit used up in a few hours...)

### 2.2 Ollama (free)

```{image} ../images/ollama.png
:alt: ollama
:width: 50%
```

- Accessible via PC/laptop software
- **Donwload URL**: [https://ollama.com](https://ollama.com)
- Open source/open weight models
- Free to use

---

## 3. Finding the Best Model for Your Task

> How do I know which model(s) is best for what task?

<img src="https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3djkxamhiZ3FmZDMzOWszY3dmdTJlbTM4eHM1MjAxa2ltdHZzZWI3biZlcD12MV9naWZzX3JlbGF0ZWQmY3Q9Zw/143ujxyRoVMJVK/giphy.gif" alt="dragon" style="width: 80%; height: auto;">



### 3.1 Chatbot Arena


[LMArena](https://lmarena.ai/) is a battle field for different large language generative models. Everytime you prompt a question, 2 anonymous AI model will be asked to answer your question, and you will vote which model gave better response. After you give your vote, the system will reveal the models' names. After millions of pairwise comparisons are made, an Elo-rating algorithm is used to rank the AI models based on people's preferences.

```{image} ../images/lmarena.png
:alt: arena
:width: 100%
```

[Text-to-image arena](https://lmarena.ai/?chat-modality=image) uses the same logic, but it is a battle field for different text-to-image generative models.

- **Free to try all models**
- You can vote and contribute to rankings!
- Great for comparing model performance


```{image} ../images/llm_rank.png
:alt: rank
:width: 100%
```

---

## 4. Using Proprietary Models Economically

**API (Application Programming Interface)**

- Pay-per-use model
- More cost-effective for specific tasks
- Programmatic access to models

```{image} ../images/api.png
:alt: api
:width: 100%
```

---

## 5. AI Agents for Coding

### 5.1 Cursor: IDE-Based Solutions

*💡 What is Cursor IDE?**
[Cursor](https://cursor.com/agents) is an advanced code editor where AI agents helps you write code in real-time! It's like having a super-smart coding partner sitting next to you.


```{image} ../images/cursor.png
:alt: cursor
:width: 100%
```

- **Cost**: $20-200 USD/month
- **Setup**: VS Code + LLM of your choice + tools like MCP (Model Context Protocol)
- **Features**:
  - Understands your entire codebase & project
  - Popular among developers and start-up companies
  - Fast for prototyping

#### 5.1.1 🤖 Real-World Example: Cursor's Bugbot in Action

```{image} ../images/bugbot.png
:alt: cursor bugbot
:width: 60%
```

**Personal Experience**: I use Cursor's **bugbot feature** that automatically checks my newly edited code in Pull Requests (PRs). 

🐛 The bugbot actually **spotted a bug that I did not catch after running 100+ tests!** This is a perfect example of how AI can serve as an additional safety net - even when your code passes all tests, AI can still catch logical errors, edge cases, or potential issues that traditional testing might miss.

#### 5.1.2 🎥 Demo: System Prompts + GitHub Pages Tutorial

**💡 What is a System Prompt?**
A system prompt is a piece of text that set context, define the agents' persona, and guide the agents' behaviours, before it starts helping you. It's like telling someone "You are now a cooking teacher, only allowed to answer questions related to cooking" before asking cooking questions!

🎯 **The Power of System Prompts** 

AI can get lost in the woods with too much information it reads all over the internet. System prompts allow the AI agent to focus on solving specific tasks with best practices.

🧭 System prompts allow you to be the guide for AI - you become the navigator! 

**💡 Pro Tip:**
- When you learn a new concept or workflow, write notes for yourself
- The note you write can be turned into system prompts for AI
- This highlights the importance of **communication skills with humans AND with AI** → prompt engineering! 

**Watch the Demo**: [System Prompt Guided Web Development](https://youtu.be/Nt_WPfcwYZA) 

In this demo video, I show how to use **system prompts to guide Cursor's AI agent** to create a website hosted on GitHub Pages. The demo uses **Jupyter Book as a template** to create the tutorial page you're reviewing right now! 

This demonstrates the power of combining:
- 🎯 **Clear system prompts** (you being the guide)
- 🤖 **AI coding assistance** (Cursor as your copilot)  
- 🔗 **GitHub integration** (via GitHub MCP)

#### 5.1.3 ⚠️ The "Vibe Coding" Reality Check

On the internet, you can see many people start building cool games/websites **without any coding background** - they're just "vibe coding," talking to Cursor or Claude Code in human language. 💬

**The Initial Illusion**: 🏃‍♂️💨
From my experience, it could seem like they are **running very fast in the first place**.

**The Reality Wall**: 🧱
But if they don't have the fundamentals, **they can hit the wall very quickly too**:
- AI could get stuck in circles when fixing bugs 🔄
- Progress will slow down significantly 🐌
- The product will break apart in the long run if you purely depend on vibe coding 💥

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbTlrM2U4NmJvODRqZDJ6ZWF4ZWk3Y2J3eHFkajZzMW5wYnUwdGMxMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/X7jjWeC03QDT2/giphy.gif" alt="snail walking" style="width: 100%; height: auto;">

**🎯 You Do NOT Want to Be That Person**

No matter which AI agent you will be using in the future, if you take the time to:
- **Slow down** 🐢
- **Learn the fundamentals** 📚
- **Become an expert yourself** 🧠

AI tools will **shine hundreds of times brighter** ✨ when collaborating with you, compared to someone with shaking fundamentals.


### 5.2 Claude Code: Command Line Solutions

```{image} ../images/cc.png
:alt: claude code
:width: 100%
```

- **Website**: [https://docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code)
- **Cost**: $17-200 USD/month
- **Status**: Currently the best AI agent for coding
- **Note**: Use it for fun

### 5.3 Other AI Coding Agents (Prices in USD/month)

#### 5.3.1 VS Code Extensions

- **Augment** - $50 - [https://www.augmentcode.com](https://www.augmentcode.com)
- **GitHub Copilot** - Free to start
- **Cline** - Pay by token - [https://github.com/cline/cline](https://github.com/cline/cline)

#### 5.3.2 IDEs (integrated development environment)

- **Amazon Kiro** - Free (waitlist) - [https://kiro.dev/](https://kiro.dev/)
- **Windsurf** - $15 - [https://windsurf.com/editor](https://windsurf.com/editor)
- **Trae** - $3-10 - [https://www.trae.ai/](https://www.trae.ai/)
- **Replit** - $20 - [https://replit.com](https://replit.com)

#### 5.3.3 Command Line Tools

- **Google Gemini CLI** - Mostly free - [https://cloud.google.com/gemini/docs/codeassist/gemini-cli](https://cloud.google.com/gemini/docs/codeassist/gemini-cli)
- **OpenAI Codex CLI** - Pay by token - [https://openai.com/index/openai-codex/](https://openai.com/index/openai-codex/)
- **Cursor CLI** - $20-200 - [https://cursor.com/cli](https://cursor.com/cli)


---

## 6. AI for Reading

### 6.1 NotebookLM

**💡 What is NotebookLM?**
[NotebookLM](https://notebooklm.google.com/) is Google's AI-powered research assistant that can read your documents, notes, and sources to create summaries, answer questions, and even generate study guides!

**🎈 Amazing Features:**
- **Document Chat:** Ask questions about specific documents that you upload(PDFs, text files, websites, and more) you've uploaded
- **Source Synthesis:** Knows all your sources and can connect ideas across them. Combines information from multiple sources
- **Study Guide Creation:** Automatically creates organized notes, mind maps, and summaries
- **Audio Overviews:** Generates engaging podcast conversations about your content!
- - **Video Overviews:** Creates a video presentation about the document you uploaded
- **Free to use**

Below is an example usecase of notebookLM: [a scientific review paper about how dairy cows change their behaviours when they are sick](https://notebooklm.google.com/notebook/16d16956-5cad-47b0-94ba-b33a353eb26c)

```{image} ../images/notebooklm.png
:alt: notebook LM
:width: 100%
```

---

## 7. AI-Based Search Engines

*Most chatbot interface like ChatGPT, and Claude has enabled web search feature and will show you the link to source too if you ask for it.*

### 7.1 Perplexity

**💡 What is Perplexity?**
[Perplexity](https://www.perplexity.ai/) is like a LLM-powered search engine that answers your questions by summarizing across multiple webpages, with citation provided.

**🔍 How is Perplexity Different from Google?**
- **Google:** Shows you links to websites
- **Perplexity:** Reads those websites and summarizes the answer for you, provides sources and citations.

Perplexity subscriptions:
- **Free tier**: 3 questions per day
- **Pro version**: $20 USD/month for unlimited questions


Below is an example usecase of perplexity: [https://www.perplexity.ai/search/you-are-an-expert-software-eng-RPxbHEJKTzO064CJ86yZOA](https://www.perplexity.ai/search/you-are-an-expert-software-eng-RPxbHEJKTzO064CJ86yZOA)

```{image} ../images/perplexity_screenshot.png
:alt: perplexity demo
:width: 100%
```

---

![Code Order Magic](https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3c2ZmZXRwbHhsdmhuazU5Ym9xZGNwcWhibHFka2Z5eG9lOHJpcWhxYyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/hEIuLmpW9DmGA/giphy.gif)

