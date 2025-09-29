# Genetic Algorithm Summarization (DistilBART-CNN)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Unlicensed-lightgrey.svg)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational.svg)]()

A minimal research prototype that evaluates a pretrained DistilBART-CNN summarization model on CNN/DailyMail and applies a simple genetic algorithm over selected parameters to optimize ROUGE/METEOR metrics on a small validation subset. The script also prints an example article, reference summary, and generated summary for quick inspection. [attached_file:2]

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Expected outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [License](#license)
- [Citation](#citation)

## Overview
This project loads CNN/DailyMail v3.0.0, selects 100 validation examples, tokenizes inputs for DistilBART-CNN, and evaluates using ROUGE and METEOR as fitness signals while a basic genetic algorithm explores small weight perturbations on targeted parameters. It uses mixed-precision inference where available to speed up generation. [attached_file:2]

## Features
- DistilBART-CNN summarization with beam search generation and max summary length of 128 tokens. [attached_file:2]
- Fitness via ROUGE-L and METEOR using the evaluate library for consistent metrics. [attached_file:2]
- Fast experimentation: 100-example subset from the validation split to keep runtimes manageable. [attached_file:2]
- Simple GA: tournament selection, uniform crossover (50% mask), and Gaussian mutation with configurable rate. [attached_file:2]
- Mixed precision with torch.cuda.amp.autocast for GPU acceleration when available. [attached_file:2]

## Requirements
Install dependencies from requirements.txt:
- torch
- transformers
- datasets
- evaluate
- rouge_score
- numpy
[attached_file:1]

## Installation
1) Create a virtual environment (recommended):
- python -m venv .venv
- source .venv/bin/activate    # macOS/Linux
- .venv\Scripts\activate       # Windows PowerShell
[attached_file:2]

2) Install dependencies:
- pip install -r requirements.txt
Note: For GPU support, install a CUDA-compatible PyTorch wheel as per PyTorch’s instructions before installing the rest. [attached_file:1][attached_file:2]

## Quick start
Run the script from the project root:
- python GeneticAlgo.py
This will download the dataset and model on first run, evaluate baseline performance, run a small GA (population=2, generations=3, mutation_rate=0.1), and then print final metrics and an example summary. [attached_file:2]

## Configuration
Key parameters to adjust in GeneticAlgo.py:
- model_checkpoint: defaults to "sshleifer/distilbart-cnn-12-6"; replace with another Seq2Seq summarizer if desired. [attached_file:2]
- Dataset size: small_dataset = dataset["validation"].select(range(100)) — increase for more robust fitness signals. [attached_file:2]
- GA hyperparameters: population_size, num_generations, mutation_rate in GeneticAlgorithm(...) initialization. [attached_file:2]
- Generation params: max_length, num_beams in model.generate to balance quality vs. speed. [attached_file:2]
- Parameter filter: create_individual currently targets names containing "classifier". For DistilBART, change this to the correct head (e.g., "lm_head") to ensure weights are actually perturbed. [attached_file:2]

## How it works
- Loads CNN/DailyMail 3.0.0 with an increased download timeout to avoid read timeouts. [attached_file:2]
- Tokenizes articles to max_length=1024 consistent with DistilBART-CNN input constraints. [attached_file:2]
- fitness_function generates summaries and computes ROUGE and METEOR on the subset, returning a dict with "rouge" and "meteor" keys. [attached_file:2]
- GeneticAlgorithm:
  - Initializes a population as dicts mapping parameter names to tensors for the targeted subset. [attached_file:2]
  - select_parents uses tournament selection (size
