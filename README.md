Genetic Algorithm for Summarization (DistilBART-CNN)
Quick start
Run the script to download CNN/DailyMail, evaluate a pretrained DistilBART-CNN model, and optimize select weights via a simple genetic algorithm using ROUGE and METEOR as fitness signals.

Create and activate a virtual environment, then install dependencies.

Execute the Python script; it will sample 100 validation examples, iterate GA for a few generations, and print metrics plus an example summary.

Features
Uses DistilBART-CNN (sshleifer/distilbart-cnn-12-6) for abstractive summarization on CNN/DailyMail 3.0.0.

Fitness combines ROUGE-L and METEOR computed via Hugging Face evaluate package.

Small validation subset (100 examples) for fast experimentation and demonstration.

Mixed precision generation with torch.cuda.amp.autocast for speed on GPU when available.

Simple GA with tournament selection, uniform crossover, and Gaussian mutation over a targeted parameter subset placeholder (“classifier” filter).

Environment setup
Python 3.9+ recommended; GPU optional but supported via CUDA if available.

Dependencies (from requirements.txt):

torch, transformers, datasets, evaluate, rouge_score, numpy.

Install:

pip install -r requirements.txt.

If using CUDA, install a CUDA-matched PyTorch build per PyTorch guidance before other packages to ensure GPU use is available to torch.

How it works
Loads CNN/DailyMail v3.0.0 with a longer download timeout to avoid ReadTimeoutError.

Selects 100 validation samples and tokenizes to max_length 1024 for DistilBART-CNN.

Fitness function:

Generates summaries with num_beams=4 and max_length=128 under autocast.

Computes ROUGE and METEOR using evaluate.load("rouge") and evaluate.load("meteor").

GeneticAlgorithm class:

Population initialization creates individuals as dicts of tensors for parameters whose names match a filter (“classifier” placeholder).

Mutation adds Gaussian noise scaled by 0.1 with probability = mutation_rate per tensor.

Crossover applies a random 50% mask per tensor to mix parents.

Parent selection via tournament (size 3) using ROUGE as the selection metric.

Evaluation loads the individual tensors into the model and recomputes metrics via fitness_function.

After GA optimization, the best individual is loaded and the model is evaluated again; an example article/reference/generated summary triplet is printed.

Usage
Run:

python GeneticAlgo.py.

What happens:

Downloads dataset (first run), tokenizer, and model.

Runs 2-population, 3-generation GA with 0.1 mutation rate by default.

Prints per-generation best ROUGE/METEOR and final metrics, plus one example summary comparison.

Configuration
Adjust these variables in GeneticAlgo.py:

model_checkpoint: change to another Seq2Seq model compatible with summarization.

small_dataset size: increase beyond 100 for more robust fitness signals at the cost of time.

GA hyperparameters: population_size, num_generations, mutation_rate in the GeneticAlgorithm initialization.

Generation parameters: max_length, num_beams in model.generate.

Notes:

The “classifier” filter in create_individual is a placeholder; DistilBART may not have parameters with that substring. Update the filter to target specific named parameters (e.g., final linear/LM head) to actually evolve weights.

Overwriting model parameters in-place for evaluation is intentional here but expensive; consider cloning model state_dict and restoring between evaluations for safety in larger experiments.

Expected outputs
Console logs:

Input ID debug lines for tokenization and shapes during fitness evaluation.

Per-generation best fitness dict containing rouge and meteor fields.

Final model performance dict.

Example article text, reference summary, and generated summary.

Troubleshooting
Slow runs or timeouts when downloading datasets:

A longer DownloadConfig(timeout=60) is already set; consider caching datasets or increasing timeout if network is slow.

CUDA not used:

Ensure torch detects GPU and proper CUDA build is installed; the script auto-selects device = "cuda" if available.

No parameter updates:

If no model parameter names match the “classifier” filter, GA won’t actually change weights; adjust the filter to the correct head names in DistilBART (e.g., “lm_head”).

High runtime:

Reduce small_dataset size, generations, or beams; or disable debug prints for Input IDs.

Project structure
GeneticAlgo.py — main script with dataset loading, fitness function, GA class, and demo generation.

requirements.txt — Python dependencies for runtime and evaluation.

License
No license file is provided; consider adding an open-source license if distribution is intended.

Citation
Model: sshleifer/distilbart-cnn-12-6 via Hugging Face Transformers.

Dataset: CNN/DailyMail 3.0.0 via Hugging Face Datasets.

Metrics: evaluate with ROUGE and METEOR modules.
