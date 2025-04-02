
import argparse
import os
import json
from datasets import load_dataset
import ast
from prompt import *
from openai_utils import get_function_completion
import time

def json_serialize_safe(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: json_serialize_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_safe(item) for item in obj]
    else:
        return obj

def extract_function_info(function_str):
    try:
        module = ast.parse(function_str)
        function_definition = module.body[0]  
        function_name = function_definition.name
        function_args = [arg.arg for arg in function_definition.args.args]
    except Exception as e:
        # print("extract_function_info error:", e)
        return None, None, None

    if 'table' in function_args:
        function_args.remove('table')

    properties = {
        arg: {
            "type": "string",
            "description": "the thing user wants to query, maybe a person's name, a city's name, etc.",
        }
        for arg in function_args
    }
    required = [key for key in properties.keys()]
    p = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": "Use this function to answer user's questions.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
    }
    function = [p]

    return function, function_name, function_args

def create_function_from_string(function_name, function_string):
    exec(function_string)
    return locals()[function_name]

def execute_function_call(sub_question, table_data, function_extract_response, model):
    success = False
    result = None
    feedback = None
    function, function_name, function_args = extract_function_info(function_extract_response)
    if function:
        arguments = {}
        if function_args != []:
            messages = [{"role": "user", "content": sub_question}]
            function_response = get_function_completion(messages, functions=function, model=model)
            arguments = json.loads(function_response.function.arguments)
        else:
            function_response = None
        try:
            function_ref = create_function_from_string(function_name, function_extract_response)
            args = [table_data] + [arguments[i] for i in function_args]
            result = function_ref(*args)
            feedback = "Feedback: the python script is correct, nothing to fix."
            success = True
            # print("execute_function_call success, the result is:", result)
        except Exception as e:
            feedback = f"Feedback: {e}"
            # print("execute_function_call error:", e)
    return success, result, feedback

def function_call(log_data, sub_question, table_data, function_extract_response, model):
    success, result, feedback = execute_function_call(sub_question, table_data, function_extract_response, model)
    iter_num = 0
    while(not success and iter_num < 3):
        iter_num += 1
        function_extract_response = self_debugging(sub_question, table_data, function_extract_response, feedback, model)
        success, result, feedback = execute_function_call(sub_question, table_data, function_extract_response, model)
        log_data["function"].append(function_extract_response)
        if success:
            break
    if result == None or result == "None":
        # print("can not find answer by function call!")
        result = ask_directly(sub_question, table_data, model)

    # print("="*100)
    # print(f"Function Call")
    # print("-"*100)
    # print(f"Short Answer: {result}")

    return result

def process_sub_question(sub_question, table_data, model):
    # print("="*100)
    # print(f"Process Sub Question")
    # print("-"*100)
    # print(f"Sub Question: {sub_question}")

    log_data = {}
    function_response = function_generator(sub_question, table_data, model)
    function_extract_response = function_extraction(function_response, model)

    # print("="*100)
    # print(f"Function Extraction")
    # print("-"*100)
    # print(f"Function Extract Response: {function_extract_response}")

    log_data["function"] = [function_extract_response]
    short_answer = function_call(log_data, sub_question, table_data, function_extract_response, model)
    long_answer = sentence_generator(short_answer, sub_question, model)
    log_data["short_answer"] = short_answer
    log_data["long_answer"] = long_answer

    return long_answer, log_data

def get_table_answer(test_data, done_samples, n_samples, model, output_path, dataset_name):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open file in append mode to continue writing without overwriting existing data
    with open(output_path, "a", encoding="utf-8") as f:
        i = 0
        for item in test_data:
            example_id = item["example_id"]
            query = item["query"]
            if example_id in done_samples.keys():
                continue
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing #{i}...")
            i += 1
            try:
                table_data = item["table"]
                ground_truth = item["summary"]
                iter_num = 0
                log_data = []
                prediction = "error"
                old_plan = None
                while(True):
                    iter_num += 1
                    iter_data = {"iter_num": iter_num}
                    question_list = plan_generation(query, old_plan, model)
                    old_plan = question_list
                    iter_data["plan"] = question_list
                    answer_list = []
                    sub_log_list = []
                    for sub_question in question_list:
                        sub_answer, sub_log_data = process_sub_question(sub_question, table_data, model)
                        answer_list.append(sub_answer)
                        sub_log_list.append(sub_log_data)
                    iter_data["reasoning_log"] = sub_log_list
                    log_data.append(iter_data)
                    done = check_plan(query, question_list, model)
                    if done or iter_num >= 3:
                        if dataset_name == "FeTaQA":
                            prediction = generate_final_answer_fetaqa(query, answer_list, model)
                        elif dataset_name == "QTSumm":
                            prediction = generate_final_answer_qtsumm(query, answer_list, model)
                        break
                result_item = {"example_id": example_id, "query": query, "prediction": prediction, "ground_truth": ground_truth, "log_data": json_serialize_safe(log_data)}
                # print("-"*100)
                # print("Ground Truth:", ground_truth)
            except Exception as e:
                result_item = {"example_id": example_id, "query": query, "prediction": "error"}
            
            # Write current result immediately
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f.flush()  # Force flush buffer to ensure data is written to disk immediately
            print(f"✓ Result written: {example_id}")
            
            if n_samples != -1 and i >= n_samples:
                break

def clean_error_entries(output_path):
    """
    Clean error entries from output file, keeping only successful results
    """
    if not os.path.exists(output_path):
        return {}, 0, 0
    
    print(f"Cleaning error entries from file: {output_path}")
    
    # Read all data
    all_items = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                all_items.append(item)
    
    # Filter out error entries and count statistics
    valid_items = []
    done_samples = {}
    error_count = 0
    success_count = 0
    
    for item in all_items:
        if item["prediction"] != "error":
            valid_items.append(item)
            done_samples[item["example_id"]] = item
            success_count += 1
        else:
            error_count += 1
    
    # Rewrite file with only successful entries
    with open(output_path, "w", encoding="utf-8") as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Cleanup completed: kept {success_count} successful items, removed {error_count} error items")
    return done_samples, success_count, error_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-35-turbo")
    parser.add_argument("--n_samples", type=int, default=10) # -1 for all samples
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/QTSumm")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="outputs")
    args = parser.parse_args()
    model = args.model
    dataset_name = args.dataset_name.split("/")[-1]
    output_path = os.path.join(args.output_path, f"{dataset_name}_output", f"{dataset_name}_{args.split_name}_{model}_output.jsonl")
    
    # Clean error entries and get completed samples
    done_samples, success_count, error_count = clean_error_entries(output_path)
    
    test_data = load_dataset(args.dataset_name, split=args.split_name) 
    if dataset_name == "FeTaQA":
        def transform_fetaqa_to_qtsumm(example):
            table_dict = {
                'header': example['table_array'][0],
                'rows': example['table_array'][1:],
                'title': f"{example['table_page_title']}, {example['table_section_title']}"
            }
            return {
                'example_id': str(example['feta_id']),
                'query': example['question'],
                'summary': example['answer'],
                'table': table_dict
            }
        test_data = test_data.map(
            transform_fetaqa_to_qtsumm,
            remove_columns=test_data.column_names
        )
    
    print(f"Starting data processing, {success_count} samples completed, processing remaining samples...")
    
    # Process data and write in real-time
    get_table_answer(test_data, done_samples, args.n_samples, model, output_path, dataset_name)
    print("✓ All data processing completed!")
