<!--
EmbedDiff is a deep learning pipeline for de novo protein design using latent diffusion models and ESM2 embeddings. It generates novel, biologically plausible protein sequences and includes decoding, BLAST validation, entropy filtering, and structure prediction using ESMFold or AlphaFold2. Ideal for machine learning in bioinformatics, protein engineering, and generative biology.
-->
# 🧬 EmbedDiff: Latent Diffusion Pipeline for De Novo Protein Sequence Generation

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Run EmbedDiff](https://img.shields.io/badge/🚀-Run%20Pipeline-blue)](#-quick-start-1-liner)
[![HTML Report](https://img.shields.io/badge/View%20Report-📊-orange)](https://mgarsamo.github.io/EmbedDiff/embeddiff_summary_report.html)


**EmbedDiff** is a modular pipeline for **de novo protein sequence generation** that combines pretrained ESM2 embeddings, a latent diffusion model, and Transformer-based decoding. It enables efficient exploration of the protein sequence landscape—generating novel sequences that preserve **evolutionary plausibility**, **functional diversity**, and **foldability**, without requiring structural supervision.

---

## 🚀 Quick Start (1-liner)

To run the entire EmbedDiff pipeline from end to end:

```bash
python run_embeddiff_pipeline.py
```

## 🔍 What Is EmbedDiff?

**EmbedDiff** is a generative machine learning pipeline for de novo protein design that combines powerful pretrained embeddings with a latent diffusion model and Transformer decoding.

It starts by embedding natural protein sequences using [**ESM2**](https://github.com/facebookresearch/esm), which maps each sequence into a high-dimensional vector that encodes rich evolutionary, functional, and structural priors. These embeddings serve as a biologically grounded latent space. A **denoising diffusion model** is then trained directly on these ESM2 embeddings. During training, Gaussian noise is added to the embeddings across a series of timesteps, and the model learns to reverse this corruption—effectively modeling the distribution of natural protein embeddings. This enables EmbedDiff to sample entirely new latent vectors from noise that remain within the manifold of plausible protein sequences. These synthetic embeddings are decoded into amino acid sequences using a Transformer model, which supports both stochastic sampling and optional reference-guided decoding. The resulting sequences are novel yet biologically grounded. The pipeline concludes with comprehensive validation and visualization, including:

- **Shannon entropy filtering** to assess compositional diversity
- **BLAST alignment** against SwissProt to measure sequence novelty and identity
- **Cosine similarity** comparisons in latent space
- **t-SNE and MDS** plots for embedding visualization
- **Optional structural assessment** using [**ESMFold**](https://github.com/facebookresearch/esm) to predict 3D folds and per-residue confidence (pLDDT)

All results are compiled into an interactive **HTML summary report** for easy inspection and sharing.

---

## 📌 Pipeline Overview

The full EmbedDiff pipeline is modular and proceeds through the following stages:

### **Step 1: Input Dataset**
- Format: A curated FASTA file of real protein sequences (e.g., Thioredoxin reductases).
- Used as the basis for learning a latent protein representation and decoder training.

---

### **Step 2a: ESM2 Embedding**
- The curated sequences are embedded using the `esm2_t33_650M_UR50D` model.
- This transforms each protein into a 1280-dimensional latent vector.
- These embeddings capture functional and evolutionary constraints without any structural input.

---

### **Step 2b: t-SNE of Real Embeddings**
- t-SNE is applied to the real ESM2 embeddings to visualize the structure of protein space.
- Serves as a baseline to later compare generated (synthetic) embeddings.

---

### **Step 3: Train EmbedDiff Latent Diffusion Model**
- A denoising MLP learns to reverse the process of adding Gaussian noise to real ESM2 embeddings.
- Trained using a sequence of time steps (e.g., 30 steps), the model gradually denoises noisy embeddings back toward the real manifold.
- This enables sampling realistic embeddings from noise.

---

### **Step 4: Sample Synthetic Embeddings**
- Starting from pure Gaussian noise, the trained diffusion model is used to generate new latent vectors that resemble real protein embeddings.
- These latent samples are biologically plausible but unseen — representing de novo candidates.

---

### **Step 5a: Build Decoder Dataset**
- Real ESM2 embeddings are paired with their corresponding amino acid sequences.
- This dataset is used to train a decoder to translate from embedding → sequence.

---

### **Step 5b: Train Transformer Decoder**
- A Transformer model is trained to autoregressively generate amino acid sequences from input embeddings.
- Label smoothing and entropy filtering are used to improve sequence diversity and biological plausibility.
- Optionally, ESM2 logit distillation is applied to align predictions with natural residue distributions.

---

### 🔄 Step 6: Decode Synthetic Sequences

The synthetic embeddings from Step 4 are decoded into amino acid sequences using a **hybrid decoding strategy** that balances biological realism with diversity.

By default:
- **40%** of amino acid positions are generated **stochastically**, sampled from the decoder’s output distribution.
- **60%** are **reference-guided**, biased toward residues from the closest matching natural sequence.

This configuration is empirically tuned to produce sequences with approximately **50–60% sequence identity** to known proteins—striking a practical balance between novelty and plausibility.

#### 💡 Modular and Adjustable
This decoding step is fully configurable:
- Setting the stochastic ratio to **100%** yields **fully de novo sequences**, maximizing novelty but potentially reducing identity.
- Lower stochastic ratios (e.g., **20–30%**) increase similarity to natural proteins.
- The ratio can be adjusted using a configuration flag in the decoding script.

The output is a final FASTA file of decoded protein sequences, suitable for downstream validation or structural modeling.


---

### **Step 7a: t-SNE Overlay**
- A combined t-SNE plot compares the distribution of real and generated embeddings.
- Useful for assessing whether synthetic proteins fall within plausible latent regions.

---

### **Step 7b: Cosine Similarity Histogram**
- Pairwise cosine distances are computed between:
  - Natural vs. Natural sequences
  - Natural vs. generated sequences
  - Generated vs. generated sequences
- This helps evaluate diversity and proximity to known protein embeddings.

---

### 🔍 Step 7c: Entropy vs. Identity Filtering

Each decoded protein sequence is evaluated using two key metrics:

- **Shannon Entropy**: Quantifies amino acid diversity across the sequence.
  - Values typically range from **~1.0 (low diversity)** to **~4.3 (maximum diversity)**.
  - **Higher entropy values (≥ 3.5)** suggest diverse, non-repetitive sequences.
  - **Lower values (< 2.5)** may indicate low-complexity or biologically implausible repeats.

- **Sequence Identity (via BLAST)**: Measures similarity to known natural proteins.
  - This helps ensure the generated sequences remain evolutionarily grounded while still being novel.

Sequences are filtered based on configurable entropy and identity thresholds to retain those with **balanced novelty and biological relevance**. Only sequences within the target range are included in downstream analysis and structural validation.


---

### 🔍 Step 7d: Local BLAST Validation

Generated sequences are validated by aligning them against a **locally downloaded SwissProt database** using the `blastp` tool from **NCBI BLAST+**.

- Uses: `blastp` from the BLAST+ suite
- Target database: `SwissProt` (downloaded locally in FASTA format)
- Input: Decoded sequences (`decoded_embeddiff.fasta`)
- Output: A CSV summary with:
  - **Percent identity**
  - **E-value**
  - **Bit score**
  - **Alignment length**
  - **Matched SwissProt accession/description**

This step confirms that generated sequences are **evolutionarily meaningful** by evaluating their similarity to curated natural proteins.

> 📁 Output example: `data/blast_results/blast_summary_local.csv`


---

### **Step 8: HTML Summary Report**
- All visualizations, metrics, and links to output files are compiled into an interactive HTML report.
- Includes cosine plots, entropy scatter, identity histograms, and t-SNE/MDS projections.
- Allows easy inspection and sharing of results.

---

### 🧪 Optional: Structural Validation with ESMFold or AlphaFold2

Although not part of the core EmbedDiff pipeline, the generated sequences can optionally be assessed for structural plausibility using modern protein structure prediction tools:

#### 🔬 [ESMFold](https://github.com/facebookresearch/esm)
- A fast, accurate structure predictor from Meta AI, built on the ESM2 language model.
- Accepts a FASTA file of protein sequences as input and returns predicted 3D structures with per-residue confidence scores (pLDDT).
- Ideal for rapid, large-scale folding of EmbedDiff-generated sequences.

#### 🧬 [AlphaFold2](https://github.com/deepmind/alphafold)
- The state-of-the-art method from DeepMind for protein structure prediction.
- Provides highly accurate structural models and can be run locally or via cloud platforms.
- More computationally intensive, but offers best-in-class accuracy.
---
#### 🧯 Output from Structural Prediction Tools
- **3D Models** (`.pdb`) for each sequence.
- **Confidence Scores** (e.g. `pLDDT` or `PAE`) per residue.
- **Optional Visualizations** using tools like:
  - [`py3Dmol`](https://github.com/3dmol/3Dmol.js)
  - [`nglview`](https://github.com/nglviewer/nglview)

---

> 📌 These tools provide additional confidence that the generated sequences are likely to fold into stable and ordered protein structures.


## 🙏 Citation & Acknowledgment

If you use **EmbedDiff** in your research or development, please consider [starring the repo ⭐](https://github.com/mgarsamo/EmbedDiff) and linking back to it. Citations and backlinks help others find and trust this work.

---

## 📂 Project Structure
EmbedDiff/
├── README.md                       # 📘 Project overview and documentation
├── .gitignore                     # 🛑 Files/folders to exclude from version control
├── run_embeddiff_pipeline.py      # 🧠 Master pipeline script to run all steps
├── requirements.txt               # 📦 Python dependencies for setting up environment
├── environment.yml                # (Optional) Conda environment file (if using Conda)
│
├── data/                          # 📁 Input and output biological data
│   ├── curated_thioredoxin_reductase.fasta
│   ├── decoded_embeddiff.fasta
│   └── blast_results/
│       └── blast_summary_local.csv
│
├── embeddings/                    # 📁 Latent vector representations
│   ├── esm2_embeddings.npy
│   └── sampled_embeddings.npy
│
├── figures/                       # 📁 All generated plots and report
│   ├── fig2b_loss_train_val.png
│   ├── fig3a_generated_tsne.png
│   ├── fig5a_decoder_loss.png
│   ├── fig5b_identity_histogram.png
│   ├── fig5c_entropy_scatter.png
│   ├── fig5d_all_histograms.png
│   ├── fig_tsne_by_domain.png
│   ├── fig5f_tsne_domain_overlay.png
│   ├── fig5b_identity_scores.csv
│   └── embeddiff_summary_report.html
│
├── scripts/                       # 📁 Core processing scripts
│   ├── esm_embedder.py                    # Step 2a: Embed sequences with ESM2
│   ├── first_tsne_embedding.py           # Step 2b: t-SNE of real embeddings
│   ├── train_emeddiff.py                 # Step 3: Train latent diffusion model
│   ├── sample_embeddings.py              # Step 4: Sample new embeddings
│   ├── build_decoder_dataset.py          # Step 5a: Build decoder training set
│   ├── train_transformer.py              # Step 5b: Train decoder
│   ├── transformer_decode.py             # Step 6: Decode embeddings to sequences
│   ├── plot_tsne_class_overlay.py        # Step 7a: t-SNE comparison
│   ├── cosine_simlar_histo.py            # Step 7b: Cosine similarity plots
│   ├── plot_entropy_identity.py          # Step 7c: Entropy vs. identity filter
│   ├── blastlocal.py                     # Step 7d: Local BLAST alignment
│   └── generate_html_report.py           # Step 8: Generate final HTML report
│
├── models/                       # 📁 ML model architectures
│   ├── diffusion_mlp.py                  # EmbedDiff diffusion model
│   └── decoder_transformer.py           # Transformer decoder
│
├── utils/                        # 📁 Utility and helper functions
│   ├── amino_acid_utils.py               # Mapping functions for sequences
│   └── metrics.py                        # Functions for loss, entropy, identity, etc.
│
└── checkpoints/                 # 📁 Model checkpoints (excluded via .gitignore)
    ├── embeddiff_mlp.pth
    └── decoder_transformer_best.pth

