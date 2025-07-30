# Brilliant Muslim Arabic Lesson Recall

[![License: MIT](https://img.shields.io/badge/license-MIT-454377.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/read-the%20docs-4d6a91.svg)](https://github.com/aatif-dawawala/arabic-lesson-eval/wiki)
[![Streamlit Cloud](https://img.shields.io/badge/demo-Open%20App-14ae73.svg)](https://brilliant-muslim-ai-recall.streamlit.app/)

An **interactive, AI-powered recall trainer** built for learners of the *Bayyinah Institute Dream* curriculum. Type what you remember, let the app retrieve the relevant material, and get instant formative feedback.

## Table of Contents

<!-- TOC -->

* [Overview](#overview)
* [Stack](#stack)
* [Features](#features)
* [Live Demo](#live-demo)
* [Getting Started](#getting-started)

  * [Clone the Repository](#clone-the-repository)
  * [Install Dependencies](#install-dependencies)
  * [Environment Variables](#environment-variables)
  * [Run Locally](#run-locally)
* [Architecture](#architecture)
* [Contributing](#contributing)
* [License](#license)

<!-- TOC -->

## Overview

Brilliant Muslim’s **Arabic Lesson Evaluator** helps students shift from passive review to **active recall**. It embeds the Dream textbook as searchable vectors, evaluates free-text answers with large language models, and delivers color-coded feedback so you know exactly what to fix next.

## Stack

| Layer          | Tech & Services                    |
| -------------- | ---------------------------------- |
| **LLM / Eval** | OpenAI GPT-4o · Gemini 2.5 Pro     |
| **Retrieval**  | FAISS vector store + LangChain RAG |
| **Frontend**   | Streamlit                          |
| **Hosting**    | Streamlit Community Cloud (demo)   |

## Features

1. **Lesson Browser** – Flip through textbook units inside the app.
2. **Active Recall Input** – Type everything you remember; no multiple-choice crutches.
3. **RAG Retrieval** – Relevant textbook passages surface automatically.
4. **AI Scoring** – Your answer is graded on accuracy and completeness.
5. **Granular Feedback**

   * ✅ Correct points
   * ❌ Misunderstandings
   * ⚠️ Missed concepts
   * 💡 Actionable advice

## Live Demo

> **Try it now →** [https://brilliant-muslim-ai-recall.streamlit.app/](https://brilliant-muslim-ai-recall.streamlit.app/)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/aatif-dawawala/arabic-lesson-eval.git
cd arabic-lesson-eval
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` **or** `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY   = "your-openai-key"
GEMINI_API_KEY   = "your-gemini-key"
```

### Run Locally

```bash
streamlit run streamlit_app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

## Architecture

```text
┌──────────────┐
│  Streamlit   │  ←– UI
└──────┬───────┘
       │ user prompt
┌──────▼─────────┐
│ LangChain RAG  │  ←– chunking & similarity search
└──────┬─────────┘
       │ relevant docs
┌──────▼─────────┐
│   LLM Scorer   │  ←– model
└──────┬─────────┘
       │ feedback JSON
┌──────▼─────────┐
│  Streamlit UI  │  ←– highlights & scores
└────────────────┘
```

## Contributing

Pull requests are welcome—help us empower Arabic learners worldwide!
Please follow the conventional commit style and open an issue first if you plan a large change.

## License

This project is released under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

© 2025 Brilliant Muslim. By using this software you agree to our [Privacy Policy](https://brilliantmuslim.com/privacy).
