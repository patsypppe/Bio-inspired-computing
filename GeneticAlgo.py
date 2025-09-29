import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, DownloadConfig
import evaluate
from torch.cuda.amp import autocast

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset (e.g., CNN/DailyMail for summarization)
# Increase timeout to avoid ReadTimeoutError
download_config = DownloadConfig(timeout=60)
dataset = load_dataset("cnn_dailymail", "3.0.0", download_config=download_config)

# Use a smaller subset of the dataset for faster experimentation
small_dataset = dataset["validation"].select(range(100))  # Use only 100 examples

# Load the pretrained DistilBART-CNN model and tokenizer
model_checkpoint = "sshleifer/distilbart-cnn-12-6"  # Lightweight model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# Preprocess the dataset
def tokenize_function(examples):
    # Truncate and pad the input sequences to the model's max length (1024 for DistilBART-CNN)
    return tokenizer(examples["article"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = small_dataset.map(tokenize_function, batched=True)

# Define the fitness function (evaluate model performance)
def fitness_function(model):
    """
    Evaluate the model's performance on the validation set using ROUGE and METEOR.
    """
    # Load metrics
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # Generate summaries and compute ROUGE and METEOR
    predictions = []
    references = []
    for example in tokenized_dataset:
        input_ids = tokenizer(example["article"], return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
        
        # Debug: Check input token IDs
        print(f"Input IDs: {input_ids}")
        print(f"Input IDs shape: {input_ids.shape}")

        with autocast():  # Enable mixed precision
            summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(summary)
        references.append(example["highlights"])  # Use the actual reference summary

    # Compute ROUGE and METEOR scores
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    meteor = meteor_metric.compute(predictions=predictions, references=references)

    return {
        "rouge": rouge["rougeL"],
        "meteor": meteor["meteor"],
    }

# Define the Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, num_generations, mutation_rate, model):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.model = model
        self.weights_shape = {name: param.shape for name, param in model.named_parameters()}

    def create_individual(self):
        """
        Create a new individual (set of model weights).
        """
        individual = {}
        for name, param in self.model.named_parameters():
            if "classifier" in name:  # Only optimize the classifier head
                individual[name] = torch.randn_like(param)
        return individual

    def mutate(self, individual):
        """
        Mutate an individual's weights.
        """
        for name in individual:
            if random.random() < self.mutation_rate:
                individual[name] += torch.randn_like(individual[name]) * 0.1  # Small random perturbation
        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create two children.
        """
        child1, child2 = {}, {}
        for name in parent1:
            mask = torch.rand_like(parent1[name]) < 0.5  # Random mask for crossover
            child1[name] = torch.where(mask, parent1[name], parent2[name])
            child2[name] = torch.where(mask, parent2[name], parent1[name])
        return child1, child2

    def select_parents(self, population, fitness):
        """
        Select two parents based on fitness (tournament selection).
        """
        tournament_size = 3
        selected = []
        for _ in range(2):
            candidates = random.sample(list(zip(population, fitness)), tournament_size)
            selected.append(max(candidates, key=lambda x: x[1]["rouge"])[0])  # Select based on ROUGE score
        return selected

    def optimize(self):
        """
        Run the genetic algorithm to optimize model weights.
        """
        # Initialize the population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness = [self.evaluate_individual(individual) for individual in population]

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}/{self.num_generations}")
            new_population = []

            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness)

                # Perform crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Evaluate children
                child1_fitness = self.evaluate_individual(child1)
                child2_fitness = self.evaluate_individual(child2)

                # Add children to the new population
                new_population.extend([(child1, child1_fitness), (child2, child2_fitness)])

            # Replace the old population with the new one
            population = [individual for individual, _ in new_population]
            fitness = [fitness for _, fitness in new_population]

            # Print the best fitness in the current generation
            best_fitness = max(fitness, key=lambda x: x["rouge"])
            print(f"Best fitness in generation {generation + 1}: {best_fitness}")

        # Return the best individual
        best_index = fitness.index(max(fitness, key=lambda x: x["rouge"]))
        return population[best_index]

    def evaluate_individual(self, individual):
        """
        Evaluate an individual's fitness by loading its weights into the model.
        """
        # Load the individual's weights into the model
        for name, param in self.model.named_parameters():
            if name in individual:
                param.data = individual[name].to(device)

        # Evaluate the model
        metrics = fitness_function(self.model)
        return metrics

# Run the Genetic Algorithm
ga = GeneticAlgorithm(population_size=2, num_generations=3, mutation_rate=0.1, model=model)
best_individual = ga.optimize()

# Load the best individual's weights into the model
for name, param in model.named_parameters():
    if name in best_individual:
        param.data = best_individual[name].to(device)

# Evaluate the final model
final_metrics = fitness_function(model)
print(f"Final model performance: {final_metrics}")

# Generate example summaries
def generate_summary(model, article):
    input_ids = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
    with autocast():  # Enable mixed precision
        summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example article and reference summary
example_article = small_dataset[0]["article"]
reference_summary = small_dataset[0]["highlights"]

# Generate summary using the best model
generated_summary = generate_summary(model, example_article)

# Print results
print("\nExample Article:")
print(example_article)
print("\nReference Summary:")
print(reference_summary)
print("\nGenerated Summary:")
print(generated_summary)