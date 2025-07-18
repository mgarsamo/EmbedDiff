# 🧬 EmbedDiff: Latent Diffusion for Protein Sequence Generation

**EmbedDiff** is a protein sequence generation pipeline that combines large-scale pretrained protein embeddings with a latent diffusion model to explore and sample from the vast protein sequence space. It generates novel sequences that preserve semantic and evolutionary properties without relying on explicit structural data, and evaluates them through a suite of biologically meaningful analyses.

---

## 🔍 What Is EmbedDiff?

EmbedDiff uses ESM2 (Evolutionary Scale Modeling v2) to project protein sequences into a high-dimensional latent space rich in evolutionary and functional priors. A denoising latent diffusion model is trained to learn the distribution of these embeddings and generate new ones from random noise. These latent vectors represent plausible protein-like states and are decoded into sequences using a Transformer decoder that blends both stochastic and reference-guided sampling.

The pipeline concludes with sequence validation via entropy, cosine similarity, BLAST alignment, and embedding visualization (t-SNE, MDS). A final HTML report presents all figures and results in an interactive format.

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
  - Higher entropy values indicate more diverse residue composition, which is often associated with novel and realistic sequences.  
  - Lower entropy values may indicate repetitive or biologically implausible sequences.

- **Sequence Identity (via BLAST)**: Measures similarity to known natural proteins.  
  - This ensures generated sequences are evolutionarily plausible while avoiding exact duplication of existing sequences.

Sequences are filtered based on configurable entropy and identity thresholds to strike a balance between **novelty** and **biological relevance**. Only sequences within the desired range are retained for downstream analysis.


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

## 📂 Project Structure
EmbedDiff/
├── README.md                       # 📘 Project overview and documentation
├── .gitignore                     # 🛑 Files/folders to exclude from version control
├── master.py                      # 🧠 Master pipeline script to run all steps
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
