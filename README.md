
# 🧠 Fine-Tuning LLaMA for Medical QA using QLoRA

This repository demonstrates how to fine-tune Meta's [LLaMA-3.2-1B-Instruct](https://huggingface.co/meta-llama) using [QLoRA](https://arxiv.org/abs/2305.14314) for a medical question-answering task focused on symptom-based queries. This project adapts a large language model efficiently using low-rank adapters and 4-bit quantization, enabling training in resource-constrained environments.

---

## 🔍 Project Overview

- **Task:** Symptom-focused medical question answering
- **Model:** LLaMA-3.2-1B-Instruct + QLoRA
- **Dataset:** Custom CSV of medical questions (filtered by qtype = symptoms)
- **Training Framework:** Hugging Face `transformers`, `trl`, `peft`
- **Approach:** Prompt-completion format fine-tuning with 4-bit quantized model and LoRA adapters
- **Evaluation:** Loss, Perplexity, Confusion Matrix, Error Analysis

---

## 📁 Repository Structure

```
Fine-Tuning-LLaMA-for-Medical-QA/
├── train.csv                      # Original dataset (question, answer, qtype)
├── symptom_qa.jsonl              # Preprocessed JSONL (prompt + completion)
├── llama_symptom_finetuned/      # Fine-tuned model checkpoints
├── graphs/                       # Visualizations: training loss, confusion matrix, etc.
│   ├── training_loss.png
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_finetuned.png
│   └── learning_curve.png
├── notebook/                     # Optional Colab/Jupyter notebooks
├── LLaMA_QLoRA_Presentation.pptx # Final 5–10 min project presentation
└── README.md
```

---

## 🚀 Setup (Google Colab Instructions)

### 1. Install Dependencies

```bash
pip install transformers datasets bitsandbytes accelerate trl peft
pip install fsspec==2025.3.2
```

### 2. Authenticate with Hugging Face

```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

### 3. Prepare Dataset

```python
import pandas as pd

df = pd.read_csv("train.csv")
symptom_df = df[df["qtype"].str.lower() == "symptoms"]
symptom_df["prompt"] = "### Question Type:\nsymptoms\n\n### Question:\n" + symptom_df["Question"] + "\n\n### Answer:"
symptom_df["completion"] = symptom_df["Answer"]
symptom_df[["prompt", "completion"]].to_json("symptom_qa.jsonl", orient="records", lines=True)
```

### 4. Load and Split Dataset

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="symptom_qa.jsonl", split="train")
train_dataset, temp_dataset = dataset.train_test_split(test_size=0.2).values()
val_dataset, test_dataset = temp_dataset.train_test_split(test_size=0.5).values()
```

---

## 🔧 Fine-Tuning Strategy

- Load model in 4-bit using `BitsAndBytesConfig`
- Apply `LoRAConfig` to target `q_proj` and `v_proj`
- Use `SFTTrainer` with masking prompt tokens for label loss
- Training config:
  - Learning Rate: 1e-4
  - Batch Size: 2–4
  - Epochs: 5
  - Mixed Precision: `fp16`

---

## 🧪 Evaluation & Error Analysis

- Metrics: Final loss, Perplexity, Error Count
- Visuals:
  - Confusion Matrices (Baseline vs Fine-Tuned)
  - Error Length Histogram
  - Training Loss Curve

---

## 📊 Key Results

| Metric            | Baseline Model | Fine-Tuned Model |
|-------------------|----------------|------------------|
| Loss              | 2.1705         | **1.4230**       |
| Perplexity        | 8.76           | **4.15**         |

---

## 📈 Visualizations

- `training_loss.png` – Training loss vs steps
- `confusion_matrix_baseline.png`
- `confusion_matrix_finetuned.png`
- `learning_curve.png` – Optional: final loss comparison

---

## 💡 Lessons Learned

- QLoRA is extremely efficient on low-resource machines
- Prompt formatting has a huge impact on output quality
- Confusion matrices and token-level error histograms provide better interpretability
- Soft evaluation is more realistic than exact string matching

---

## 🛠 Future Improvements

- Extend support to diagnosis/treatment Q&A
- Deploy as a chatbot using Streamlit or Gradio
- Add semantic/fuzzy evaluation metrics
- Integrate RAG for fact-based grounding

---

## 📬 Contact

**Satish Mallikarjun Paraddi**  
📧 paraddi.s@northeastern.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/satish-mallikarjun-paraddi)
