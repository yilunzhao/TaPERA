import json
import evaluate
import nltk
import torch
from autoacu import A3CU
from typing import List
from nltk import word_tokenize
from torch.utils.data import DataLoader
from tapas_acc import TapasTest, MyData
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.tapas.tokenization_tapas")

def load_data(file_path):
    predictions = []
    ground_truths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                predictions.append(str(item.get('prediction', '')))
                ground_truths.append(str(item.get('ground_truth', '')))
    return predictions, ground_truths

def get_sacrebleu_scores(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def get_rougel_scores(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results["rougeL"] * 100

def get_meteor_scores(predictions, references):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"] * 100

def get_bert_scores(predictions, references):
    bertscore = evaluate.load("bertscore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = bertscore.compute(predictions=predictions, references=references, lang="en", device=device)
    avg_f1 = sum(results["f1"]) / len(results["f1"])
    return avg_f1 * 100

def get_tapas_scores(prediction_file, dataset_name, split_name):
    tapas = TapasTest("google/tapas-large-finetuned-tabfact")
    data = MyData(prediction_file, dataset_name, split_name, tapas.tokenizer)
    test_dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=4)
    results = tapas.test(test_dataloader)
    return results["acc"] * 100
    
def get_autoacu_scores(predictions, references):
    device = 0 if torch.cuda.is_available() else -1
    print(f"AutoACU using device: {'cuda:0' if device == 0 else 'cpu'}")
    a3cu = A3CU(device=device)
    _, _, f1_scores = a3cu.score(
        references=references,
        candidates=predictions,
        batch_size=256,
        output_path=None,
    )
    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_f1 * 100

def get_prediction_lengths(predictions):
    total_length = 0
    for prediction in predictions:
        total_length += len(word_tokenize(prediction))
    return total_length / len(predictions)

def run_full_evaluation(predictions, references, prediction_file, dataset_name, split_name):
    all_scores = {}
    
    print("--- Start calculating metrics ---")

    print("Calculating sacreBLEU...")
    all_scores["sacreBLEU"] = get_sacrebleu_scores(predictions, references)
    
    print("Calculating Rouge-L...")
    all_scores["Rouge-L"] = get_rougel_scores(predictions, references)

    print("Calculating METEOR...")
    all_scores["METEOR"] = get_meteor_scores(predictions, references)

    print("Calculating BERTScore...")
    all_scores["BERTScore"] = get_bert_scores(predictions, references)
    
    print("Calculating TAPAS-Acc...")
    all_scores["TAPAS-Acc"] = get_tapas_scores(prediction_file, dataset_name, split_name)
    
    print("Calculating AutoACU...")
    all_scores["AutoACU"] = get_autoacu_scores(predictions, references)
    
    print("Calculating Prediction Length...")
    all_scores["Prediction Length"] = get_prediction_lengths(predictions)

    print("--- Calculation completed ---")
    return all_scores

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' data package...")
        nltk.download('punkt')

    datasets_to_evaluate = {
        "FeTaQA": {
            "file_path": "outputs/FeTaQA_output/FeTaQA_test_gpt-35-turbo_output.jsonl",
            "dataset_name": "DongfuJiang/FeTaQA",
            "split_name": "test"
        },
        "QTSumm": {
            "file_path": "outputs/QTSumm_output/QTSumm_test_gpt-35-turbo_output.jsonl",
            "dataset_name": "yale-nlp/QTSumm",
            "split_name": "test"
        }
    }

    final_results = {}

    for name, config in datasets_to_evaluate.items():
        print(f"======================================================")
        print(f" Evaluating dataset: {name}")
        print(f"======================================================")

        print(f"Loading data from {config['file_path']}...")
        try:
            predictions, ground_truths = load_data(config['file_path'])
            print(f"Successfully loaded {len(predictions)} samples.")
        except FileNotFoundError:
            print(f"Error: File not found {config['file_path']}. Skipping this dataset.")
            continue

        scores = run_full_evaluation(
            predictions=predictions,
            references=ground_truths,
            prediction_file=config['file_path'],
            dataset_name=config['dataset_name'],
            split_name=config['split_name']
        )
        final_results[name] = scores
        
    print("======================================================")
    print(" All evaluations completed - Final results summary")
    print("======================================================")
    for dataset_name, scores in final_results.items():
        print(f"--- {dataset_name} ---")
        for metric, score in scores.items():
            if score != -1.0:
                print(f"{metric:<20} | {score:.4f}")