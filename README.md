# Converging to a Lingua Franca: Evolution of Linguistic Regions and Semantics Alignment in Multilingual Large Language Models

**Codebase for the paper**  
**"Converging to a Lingua Franca: Evolution of Linguistic Regions and Semantics Alignment in Multilingual Large Language Models"**  
Hongchuan Zeng, Senyu Han, Lu Chen†, Kai Yu†  
(†corresponding authors)

[[arXiv:2410.11718v2](https://arxiv.org/abs/2410.11718)]
[[2025.coling-main.707](https://aclanthology.org/2025.coling-main.707/)]


<p align="center">
<img src="https://github.com/X-LANCE/LinguaFranca/blob/main/linguafranca.png" width=100% height=100% 
class="center">
</p>

---

## 🌐 Overview

This project explores the emergence of **language-independent semantic spaces**—a “**Lingua Franca**”—within Multilingual Large Language Models (MLLMs). We:
- Identify **key linguistic regions** that dominate representations of each language.
- Track the **evolution of language-specific and semantic activations** across layers.
- Introduce two core metrics:
  - **LRDS** (Linguistic Region Development Score)
  - **SADS** (Semantic Alignment Development Score)
- Evaluate robustness of MLLMs through **neuron-level probing, ablation, and PPL/task-based evaluation**.

We use models like BLOOM and LLaMA2 to validate our findings on datasets such as Bible, FLORES, and XLSum.

---

## 📦 Features

- 🔍 Hook-based activation capture at neuron level.
- 🧠 Automatic detection of functinal language-specific key neurons.
- 🧪 Evaluation of semantic alignment via cosine similarity.
- 🔥 Ablation analysis on neuron sets and their impact on downstream tasks and perplexity.
- 📊 Visualization of cross-lingual similarities and neuron contributions.


---

## 🚀 Quick Start

### Install dependencies:

```bash
pip install torch transformers datasets matplotlib seaborn scikit-learn tqdm
```

### Run main experiment (example with BLOOM-7B):

```bash
python run.py \
  --sample_num 100 \
  --dataset_name "bible" \
  --model "bigscience/bloom-7b1" \
  --deactivate 1 \
  --evaluate_ppl 1 \
  --evaluate_tasks 1
```

### Arguments:
- `--sample_num`: Number of examples per language.
- `--dataset_name`: Dataset name (`bible` or `flores`).
- `--model`: HF model name or local checkpoint path.
- `--deactivate`: Whether to deactivate key neurons for ablation testing.
- `--evaluate_ppl`: Evaluate PPL on XLSum with ablation.
- `--evaluate_tasks`: Run zero-shot evaluation (e.g., XStoryCloze).
- `--revision`: Optional model revision tag.

---

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **LRDS** | Measures average pairwise similarity of hidden states grouped by language. |
| **SADS** | Measures average similarity of translations of the same meaning across languages. |
| **Z-Score Neuron Ranking** | Identifies language-specific neurons contributing most to language information. |

---

## 📌 Supported Models

- BLOOM-7B1 (`bigscience/bloom-7b1`)
- LLaMA-2 (HF-compatible paths)
- Custom LLaMA/Baichuan variants

You can add support for other models by modifying hook registration logic in `generate_hidden_states_*` and hook classes.

---

## 📂 Outputs

- Cosine similarity heatmaps (`.png`)
- Key neuron scores per layer/language (`.csv`)
- Perplexity logs per language (`.csv`)
- LM evaluation logs (`.json`)

---

## 📄 Citation

If you find this project useful, please cite:

```
@inproceedings{zeng-etal-2025-converging,
    title = "Converging to a Lingua Franca: Evolution of Linguistic Regions and Semantics Alignment in Multilingual Large Language Models",
    author = "Zeng, Hongchuan  and
      Han, Senyu  and
      Chen, Lu  and
      Yu, Kai",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.707/",
    pages = "10602--10617",
}
```

---

## 💬 Contact

For questions, contact:  
`charlie68@sjtu.edu.cn`, `chenlusz@sjtu.edu.cn`, `kai.yu@sjtu.edu.cn`
