# 💬 Finance Assistant AI

An intelligent chatbot powered by **LLM + RAG** (Retrieval-Augmented Generation) to assist with financial queries, fraud detection insights, and company-related Q&A using custom financial datasets.

---

## 🚀 Features

- 💸 Ask anything about finance or fraud detection
- 🔎 Uses vector search (FAISS) to retrieve relevant context
- 🧠 Mistral-7B-Instruct model used for answering questions
- 📊 Supports multiple datasets (Sentiment, Fraud, QA)
- 🧾 Displays relevant labels and context for transparency

---

## 🔐 Hugging Face Token Setup
To use the Mistral LLM, add your Hugging Face token:

🔗 Get your token here: https://huggingface.co/settings/tokens

Set it as an environment variable:
```bash
export HF_TOKEN=your_token_here
```


## 📂 Supported Datasets

| Dataset Name        | Purpose            | Required Columns                     |
|---------------------|--------------------|--------------------------------------|
| `finance1.csv`      | Sentiment Analysis | `text`, `label`                      |
| `synthetic_log.csv` | Fraud Detection    | `type`, `nameOrig`, `isFraud`, etc. |
| `financial_qa.csv`  | Financial Q&A      | `question`, `answer`, `context`     |


## ▶️ How to Run
Run the app with:
```bash
streamlit run app.py
```

 ## 🧰 Technologies Used
● Streamlit

● FAISS

● Hugging Face Transformers

● Sentence Transformers

● Scikit-learn

## 👩‍💻 Author
Made by Yashika Maligi
