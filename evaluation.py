import os
import pandas as pd
import torch
from transformers import AutoTokenizer
import evaluate

# Load Model & Tokenizer
def load_model_and_tokenizer(model_checkpoint="Salesforce/codet5-small"):
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_tokens(["<mask>"])
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

# Preprocess function for tokenization
def preprocess_function(examples, tokenizer):
    inputs = examples["flattened_code"]
    targets = examples["target_block"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Test function for evaluating the model's performance
def test_code(csv_file, model, tokenizer, output_path="testset-results.csv"):
    sacrebleu = evaluate.load("sacrebleu")
    predictions = []
    references = []

    flattened_input = []
    correct = []
    targets = []
    code_pred = []
    Code_Bleu = []
    Bleu_4 = []

    i = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for _, row in csv_file.iterrows():
        input = row["flattened_code"]
        target = row["target_block"]
        inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_length=256)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        equal_match = prediction == target
        flattened_input.append(input)
        code_pred.append(prediction)
        targets.append(target)
        predictions.append(prediction)
        references.append([target])
        correct.append(equal_match)

        # Temporary files for CodeBLEU calculation
        with open(f'/content/temp_ref.txt', 'w') as f_ref, open(f'/content/temp_pred.txt', 'w') as f_pred:
            f_ref.write(target + '\n')
            f_pred.write(prediction + '\n')

        # Call the calc_code_bleu script for CodeBLEU score
        result = os.popen(
            'cd /content/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/ && '
            'python calc_code_bleu.py --refs /content/temp_ref.txt '
            '--hyp /content/temp_pred.txt --lang java --params 0.25,0.25,0.25,0.25'
        ).read()

        # Extract CodeBLEU score
        for line in result.split('\n'):
            if 'CodeBLEU score' in line:
                score_str = str(float(line.split(':')[-1].strip()) * 100)
                Code_Bleu.append(score_str)
                break

        # Compute BLEU-4 score
        results = sacrebleu.compute(predictions=predictions, references=references)
        Bleu_4.append(round(results["score"], 6))

        if i % 10 == 0:
            print(f"This is iteration: {i}")
            print(f"The input is: {input}")
            print(f"The prediction is: {prediction}")
            print(f"The target is: {target}")
            print(f"The prediction is equal to the target? {equal_match}")
            print(f"The CodeBLEU score is: {Code_Bleu[i]}")
            print(f"The BLEU-4 score is: {Bleu_4[i]}")
            print()

        i += 1

    # Create DataFrame with results
    results_df = pd.DataFrame({
        "Input": flattened_input,
        "Exact Match": correct,
        "Expected if condition": targets,
        "Predicted if condition": code_pred,
        "CodeBLEU Score": Code_Bleu,
        "BLEU-4 Score": Bleu_4,
    })

    # Save results to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")


def main(csv_file_path, model_checkpoint="Salesforce/codet5-small", output_path="testset-results.csv"):
    
    csv_file = pd.read_csv(csv_file_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_checkpoint)

    # Test the model and save evaluation results
    test_code(csv_file, model, tokenizer, output_path)

if __name__ == "__main__":
    csv_file_path = "data/processed/flattened_test.csv"
    main(csv_file_path)