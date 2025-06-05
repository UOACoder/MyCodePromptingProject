from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from collections import defaultdict
import os

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer_path = model_id

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("tokenizer.pad_token was not set, setting it to tokenizer.eos_token")

except Exception as e:
    print(f"Failed to load tokenizer '{tokenizer_path}': {e}")
    exit()

try:
    print(f"Loading model '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.to(device)
    print(f"Model successfully loaded and transferred to {device} device")
except Exception as e:
    print(f"Failed to load model '{model_id}' or transfer to device: {e}")
    print(f"If the error is 'BFloat16 is not supported on MPS' or similar, please ensure torch_dtype is set correctly.")
    print(f"Currently attempted torch_dtype is torch.float16.")
    exit()



def load_dataset(file_path):
    """Load dataset in JSON format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Dataset file '{file_path}' is not valid JSON format.")
        return None

def llm_call(prompt_messages, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95):
    """Generic LLM call function"""
    try:
        # Ensure tokenizer has pad_token_id, if not, use eos_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id # Explicitly pass pad_token_id
        )
        response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response_text.strip()
    except Exception as e:
        print(f"Error occurred during LLM call: {e}")
        return "error"

def generate_code_representation_llm(given_info, question):
    """
    First stage LLM call: Generate Python code representation based on given_info and question.
    """
    prompt = f"""
Based on the provided information and question, generate a Python code representation that helps determine the answer to the question.
The Python code should:
1. Define variables, classes or functions that represent the causal relationships described in the 'given_info'
2. Model counterfactual scenarios mentioned in the 'question'
3. Include evaluative logic to determine the answer to the question
4. Be clear, concise and self-contained
5. Return a final result or conclusion

Given Information: "{given_info}"
Question: "{question}"

Important: Output ONLY the Python code without any explanation or markdown formatting.

    """
    messages = [{'role': 'user', 'content': prompt}]
    code_representation = llm_call(messages, max_new_tokens=1024)
    if "```python" in code_representation:
        code_representation = code_representation.split("```python")[1].split("```")[0].strip()
    elif "```" in code_representation:
        code_representation = code_representation.split("```")[1].split("```")[0].strip()
    return code_representation

def reason_with_code_llm(code_representation, original_question, original_info):
    """
    Second stage LLM call: Reason based on generated code representation, original question and information,
    output "yes" or "no".
    """
    prompt = f"""
You are an AI assistant tasked with logical reasoning based on counterfactual scenarios.
Analyze the following Python code representation that models the causal relationships described in the original information.
Then determine if the answer to the original question is "yes" or "no".

Python Code Representation:
```python
{code_representation}
```

Original Information: "{original_info}"
Original Question: "{original_question}"

Based on the Python code representation, the original information, and the original question, what is the answer to the original question?
IMPORTANT INSTRUCTION: Your final response MUST be EXACTLY one word - either "yes" or "no" with absolutely no other text.
    """
    messages = [{'role': 'user', 'content': prompt}]
    answer = llm_call(messages, max_new_tokens=10).lower()
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        print(f"Warning: Code-based reasoning LLM output unexpected answer: '{answer}' for question: '{original_question}'")
        return "error"

def reason_with_cot_llm(given_info, question):
    """
    Text prompt path LLM call: Uses Chain-of-Thought style text prompt based on given_info and question 
    for reasoning, outputs "yes" or "no".
    """
    prompt = f"""
Instructions:
1. Analyze the question using a detailed chain-of-thought reasoning process
2. Consider all causal relationships and conditions described in the given information
3. Break down your reasoning into clear, logical steps
4. After completing your reasoning, You MUST respond with only "yes" or "no".

Given Information: "{given_info}"
Question: "{question}"

Let's think step by step:
    """
    messages = [{'role': 'user', 'content': prompt}]
    response = llm_call(messages, max_new_tokens=768)

    if "yes" in response.lower():
        return "yes"
    elif "no" in response.lower():
        return "no"
    else:
        # Try alternate extraction if LLM just gave a simple yes/no
        # This part can be adjusted based on actual output from Mistral
        # Sometimes the model might give short yes/no after CoT
        lines = response.strip().split('\n')
        last_line_lower = lines[-1].lower().strip().replace('.', '')
        if last_line_lower == "yes":
            return "yes"
        elif last_line_lower == "no":
            return "no"
        print(f"Warning: CoT LLM output unexpected answer format: '{response}' for question: '{question}'")
        return "error"

# --- Main Experiment Logic ---
def main():
    # 1. Load model and tokenizer (already done globally)

    # 2. Load CounterBenchDataset_100.json dataset
    dataset_file = "CounterBenchDataset_100.json"
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not in current directory {os.getcwd()}. Please check file path.")
        return

    dataset = load_dataset(dataset_file)
    if dataset is None:
        return

    results = []
    correct_code_path = 0
    correct_text_path = 0
    total_processed_questions = 0

    accuracy_by_type_code = defaultdict(lambda: {'correct': 0, 'total': 0})
    accuracy_by_graph_id_code = defaultdict(lambda: {'correct': 0, 'total': 0})
    accuracy_by_type_text = defaultdict(lambda: {'correct': 0, 'total': 0})
    accuracy_by_graph_id_text = defaultdict(lambda: {'correct': 0, 'total': 0})

    print(f"Starting to process {len(dataset)} data points...")

    for i, item in enumerate(dataset):
        print(f"\nProcessing data {i+1}/{len(dataset)}: Question ID {item.get('question_id', 'N/A')} | Question TYPE {item.get('type', 'N/A')}")
        given_info = item.get("given_info", "")
        question = item.get("question", "")
        true_answer = item.get("answer", "").lower()
        question_type = item.get("type", "unknown")
        graph_id = item.get("meta", {}).get("graph_id", "unknown")

        if not given_info or not question or not true_answer:
            print(f"Warning: Data item {i+1} missing 'given_info', 'question', or 'answer'. Skipping this item.")
            continue

        current_result_entry = {
            "question_id": item.get("question_id"),
            "question": question,
            "given_info": given_info,
            "true_answer": true_answer,
            "type": question_type,
            "meta": item.get("meta"),
            "code_path_prediction": "not_processed",
            "generated_code_representation": "not_generated",
            "text_path_prediction": "not_processed",
        }
        
        valid_question_for_stats = True # Flag whether this question should be included in accuracy statistics

        # 3. For code prompt path
        print("  Code path:")
        print("    Generating code representation...")
        generated_code = generate_code_representation_llm(given_info, question)
        current_result_entry["generated_code_representation"] = generated_code

        if generated_code.lower() != "error" and generated_code.strip() != "":
            print("    Reasoning based on code...")
            code_path_prediction = reason_with_code_llm(generated_code, question, given_info)
            current_result_entry["code_path_prediction"] = code_path_prediction
            print(f"    Code path prediction: {code_path_prediction}, True answer: {true_answer}")

            if code_path_prediction != "error":
                accuracy_by_type_code[question_type]['total'] += 1
                accuracy_by_graph_id_code[graph_id]['total'] += 1
                if code_path_prediction == true_answer:
                    correct_code_path += 1
                    accuracy_by_type_code[question_type]['correct'] += 1
                    accuracy_by_graph_id_code[graph_id]['correct'] += 1
            else:
                print("    Code path reasoning returned error.")
                # valid_question_for_stats = False # If reasoning fails, can choose not to include this question in overall accuracy
        else:
            print("    Code generation returned error or empty, skipping code path reasoning.")
            current_result_entry["code_path_prediction"] = "error_code_generation"

        # 4. For text prompt path
        print("  Text path (CoT):")
        print("    Reasoning using CoT...")
        text_path_prediction = reason_with_cot_llm(given_info, question)
        current_result_entry["text_path_prediction"] = text_path_prediction
        print(f"    Text path prediction: {text_path_prediction}, True answer: {true_answer}")

        if text_path_prediction != "error":
            accuracy_by_type_text[question_type]['total'] += 1
            accuracy_by_graph_id_text[graph_id]['total'] += 1
            if text_path_prediction == true_answer:
                correct_text_path += 1
                accuracy_by_type_text[question_type]['correct'] += 1
                accuracy_by_graph_id_text[graph_id]['correct'] += 1
        else:
            print("    Text path reasoning returned error.")
            # valid_question_for_stats = False # If reasoning fails, can choose not to include this question in overall accuracy
        
        if valid_question_for_stats:
            total_processed_questions +=1

        results.append(current_result_entry)

        if (i + 1) % 10 == 0:
            temp_output_file = "data_output_Mistral7B_Reasoning.json"
            try:
                with open(temp_output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"Temporarily saved {i+1} results to {temp_output_file}")
            except Exception as e:
                print(f"Failed to save temporary results to {temp_output_file}: {e}")

    print("\n--- Experiment Results Statistics ---")
    if total_processed_questions > 0:
        # Note: The accuracy denominator here is total_processed_questions, meaning successfully processed questions not marked as invalid for statistics
        # If it wants based on all attempted questions (including errors), denominator should be len(dataset) or total valid questions after filtering
        overall_accuracy_code = (correct_code_path / accuracy_by_type_code[question_type]['total']) * 100 if accuracy_by_type_code[question_type]['total'] > 0 else 0
        overall_accuracy_text = (correct_text_path / accuracy_by_type_text[question_type]['total']) * 100 if accuracy_by_type_text[question_type]['total'] > 0 else 0
        
        # Calculate total valid processed questions for code path and text path overall accuracy
        total_valid_code_predictions = sum(stats['total'] for stats in accuracy_by_type_code.values())
        total_valid_text_predictions = sum(stats['total'] for stats in accuracy_by_type_text.values())

        overall_accuracy_code_path = (correct_code_path / total_valid_code_predictions) * 100 if total_valid_code_predictions > 0 else 0
        overall_accuracy_text_path = (correct_text_path / total_valid_text_predictions) * 100 if total_valid_text_predictions > 0 else 0

        print(f"Code path overall accuracy: {overall_accuracy_code_path:.2f}% ({correct_code_path}/{total_valid_code_predictions})")
        print(f"Text path (CoT) overall accuracy: {overall_accuracy_text_path:.2f}% ({correct_text_path}/{total_valid_text_predictions})")


        print("\nCode path - Accuracy by question type:")
        for q_type, stats in sorted(accuracy_by_type_code.items()):
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  Type '{q_type}': {acc:.2f}% ({stats['correct']}/{stats['total']})")


        print("\nText path (CoT) - Accuracy by question type:")
        for q_type, stats in sorted(accuracy_by_type_text.items()):
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  Type '{q_type}': {acc:.2f}% ({stats['correct']}/{stats['total']})")

    else:
        print("No questions were successfully processed, cannot calculate accuracy.")

    output_file = "data_output_Mistral7B_Reasoning.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nAll detailed experiment results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save final results to {output_file}: {e}")

if __name__ == "__main__":
    main()