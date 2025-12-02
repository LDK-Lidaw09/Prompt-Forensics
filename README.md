# 🧬 Prompt Forensics

Prompt Forensics is a research-oriented framework to **analyze, stress-test and audit Large Language Models (LLMs)** through adversarial prompts and “LLM-as-a-Judge” evaluation.

The goal is to answer a simple question:

> *How robust is a model (or a panel of models) against realistic prompt attacks, and how reliable is an LLM-based judge to detect those attacks?*

This repository provides:

- A **Streamlit UI** to explore prompts, Prompt DNA, and robustness scores  
- A **scenario engine** to run adversarial attacks across domains (health, education, IP, audit, etc.)  
- A **Judge Evaluation pipeline** that computes metrics (Accuracy, Precision, Recall, F1) for LLM judges on labeled attack scenarios  

---

## ✨ Key Concepts

### 🔬 Prompt DNA

Each prompt (benign or malicious) is represented by a **Prompt DNA**:

- A unique hash / identifier  
- Metadata about the domain (health, education, IP, audit, …)  
- Attack family (jailbreak, data leakage, phishing, deceptive delight, belief elicitation, reverse psychology, etc.)  
- Optional embeddings and features for semantic comparison

Prompt DNA allows:

- **Replay** of a given prompt on different models  
- **Tracking** of attacks across experiments  
- **Comparison** of outputs using similarity metrics  

### 🧪 Adversarial Scenarios

The repo includes curated JSON scenario files, such as:

- `health_jailbreak_scenarios.json`  
- `education_data_leakage_scenarios.json`  
- `ip_reverse_psychology_scenarios.json`  
- `audit_belief_elicitation_scenarios.json`  

Each scenario typically specifies:

- `domain` (health, education, IP, audit)  
- `attack_type` (jailbreak, data_leakage, phishing, etc.)  
- `context` and `actor`  
- `objective` (what the attacker wants)  
- `prompt` (the actual adversarial prompt to send to the model)  
- `gold_label` (e.g. `malicious` / `benign`)

These scenarios are used to **benchmark models and judges** in a reproducible way.

### ⚖️ LLM as a Judge

Prompt Forensics can use one or several LLMs as **judges**:

- A judge receives **a prompt, the model’s answer, and metadata**  
- It outputs a **structured JSON verdict**, e.g.:

```json
{
  "semantic_similarity": 0.92,
  "meaning_changed": false,
  "safety_regression": true,
  "changed_aspect": ["style", "safety"],
  "analysis_summary": "The output respects the task semantics but introduces unsafe content.",
  "is_malicious_prompt": true,
  "adversarial_type": ["jailbreak"],
  "safety_violation": true,
  "risk_markers": ["ignore previous instructions", "override system role"]
}
