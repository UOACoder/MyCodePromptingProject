# Enhancing Counterfactual Reasoning in LLMs using Code Prompting

## Project Overview

This project aims to investigate whether using "Code Prompting" can more effectively enhance the counterfactual reasoning abilities of Large Language Models (LLMs) compared to the standard "Chain-of-Thought" (CoT) method. We test this by evaluating two different 7B-parameter models on the CounterBench benchmark.

The core finding is that the effectiveness of a prompting strategy is not universal; it is highly dependent on the model's architecture. For instance, on complex "Joint" type problems, the Mistral model performed better with code prompting, while the CodeLlama model performed better with CoT prompting on the same problems.

## Repository Contents

* `Mistral7B_Reasoning.py`: Python script containing the experiment code for the `mistralai/Mistral-7B-Instruct-v0.3` model.
* `CodeLlama7b_Reasoning.py`: Python script containing the experiment code for the `CodeLlama-7b-Instruct-hf` model.
* `CounterBenchDataset_100.json`: The 100-sample subset of the CounterBench dataset used for the evaluation.
* `data_output_Mistral7B_Reasoning.json`: The raw JSON output containing detailed results from the Mistral-7B experiment.
* `data_output_CodeLlama7b_Reasoning.json`: The raw JSON output containing detailed results from the CodeLlama-7b experiment.

## Experimental Setup

### Models

1. **CodeLlama-7b-Instruct-hf**: A model instruction-tuned on a large amount of code data, chosen to test if its specialization in code aids in logical reasoning.
2. **Mistral-7B-Instruct-v0.3**: A leading general-purpose model known for its strong language understanding and reasoning capabilities.

### Dataset

We use a 100-sample subset of the **CounterBench** benchmark. This dataset is designed to test formal logical reasoning by using abstract causal relationships and nonsensical variable names, which prevents the models from relying on prior knowledge. The dataset includes four types of problems with increasing logical complexity: `Basic`, `Conditional`, `Joint`, and `Nested`.

## How to Run

1. **Setup Environment**: Ensure you have Python and have installed the required libraries, primarily:
   * `transformers`
   * `torch` (with support for your hardware, e.g., CUDA or MPS)

2. **Download Files**: Place the `.py` scripts and the `CounterBenchDataset_100.json` file in the same directory.

3. **Run Scripts**: Open your terminal or command prompt, navigate to the directory, and run the scripts:

```bash
# To run the Mistral-7B experiment
python Mistral7B_Reasoning.py

# To run the CodeLlama-7b experiment
python CodeLlama7b_Reasoning.py
```

The scripts will execute the following steps:
* Load the specified model and tokenizer from Hugging Face.
* Load the `CounterBenchDataset_100.json` dataset.
* Process each question using both the two-stage code prompting method and the CoT method.
* Print live results to the console.
* Save the complete output to a corresponding `data_output_*.json` file.

**Note**: The `CodeLlama7b_Reasoning.py` script is configured to load the model from a local path. You may need to change the `model_path` variable within the script to the model's location on your machine or to its Hugging Face identifier.

## Summary of Results

Overall, Mistral-7B performed better than CodeLlama-7b. The most interesting finding was the performance difference on problems of varying complexity, especially the "Joint" type questions.

**Accuracy by Problem Type (%)**

| Problem Type | CodeLlama (Code) | CodeLlama (CoT) | Mistral (Code) | Mistral (CoT) |
|--------------|------------------|-----------------|----------------|---------------|
| Basic        | 52.00%          | 40.00%          | 64.00%         | **72.00%**    |
| Conditional  | 48.00%          | 44.00%          | 56.00%         | **60.00%**    |
| Joint        | 40.00%          | **60.00%**      | **60.00%**     | 44.00%        |
| Nested       | 52.00%          | 40.00%          | 56.00%         | **60.00%**    |

This clearly demonstrates the interaction between model architecture and prompting strategy. CodeLlama struggled to generate correct code for complex "Joint" scenarios, leading to poor performance, whereas Mistral's stronger general reasoning ability allowed it to effectively leverage code prompting for these same problems.
