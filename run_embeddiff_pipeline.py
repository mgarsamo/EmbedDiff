#Master script to run the entire pipeline
import os
import subprocess
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from generate_html_report import generate_html_report  

def run_command(command, description, skip_steps, step_key):
    if step_key in skip_steps:
        print(f"⏭️ Skipping: {description}")
        return
    print(f"\n✅ {description}")
    print(f"🔧 Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Failed at: {description}")

def main():
    parser = argparse.ArgumentParser(description="Run the full EmbedDiff pipeline.")
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="List of step keys to skip (e.g. esm tsne diffusion decoder blast html)"
    )
    args = parser.parse_args()
    skip_steps = set(args.skip)

    fasta_path = "data/curated_thioredoxin_reductase.fasta"
    print(f"\n✅ Dataset prepared at: {fasta_path}")

    run_command(
        f"python utils/esm_embedder.py --input {fasta_path} --output embeddings/esm2_embeddings.npy",
        "Step 2a: ESM2 embedding of real sequences",
        skip_steps, "esm"
    )

    run_command(
        "python scripts/first_tsne_embedding.py",
        "Step 2b: Plot t-SNE of real ESM2 embeddings",
        skip_steps, "tsne"
    )

    run_command(
        "python scripts/train_emeddiff.py",
        "Step 3: Train EmbedDiff latent diffusion model",
        skip_steps, "diffusion"
    )

    run_command(
        "python scripts/sample_embeddings.py",
        "Step 4: Sample synthetic embeddings from EmbedDiff",
        skip_steps, "sample"
    )

    run_command(
        "python scripts/build_decoder_dataset.py",
        "Step 5a: Build decoder dataset from real ESM2 embeddings",
        skip_steps, "decoder_data"
    )

    run_command(
        "python scripts/train_transformer.py",
        "Step 5b: Train Transformer decoder",
        skip_steps, "decoder_train"
    )

    run_command(
        "python scripts/transformer_decode.py",
        "Step 6: Decode synthetic embeddings to amino acid sequences",
        skip_steps, "decode"
    )

    run_command(
        "python scripts/plot_tsne_class_overlay.py",
        "Step 7a: Overlay real vs. generated embeddings via t-SNE",
        skip_steps, "tsne_overlay"
    )

    run_command(
        "python scripts/cosine_simlar_histo.py",
        "Step 7b: Plot cosine similarity histogram",
        skip_steps, "cosine"
    )

    run_command(
        "python scripts/plot_entropy_identity.py",
        "Step 7c: Plot entropy vs. sequence identity",
        skip_steps, "entropy"
    )

    run_command(
        "python scripts/blastlocal.py",
        "Step 7d: Run local BLAST and summarize results (cached if available)",
        skip_steps, "blast"
    )

    # === Final HTML Report ===
    if "html" in skip_steps:
        print("⏭️ Skipping: Step 8 - Generate HTML Summary Report")
    else:
        print("\n✅ Step 8: Generate HTML Summary Report")
        generate_html_report("embeddiff_summary_report.html")

    print("\n🎉 All steps in the EmbedDiff pipeline completed successfully!")

if __name__ == "__main__":
    main()
