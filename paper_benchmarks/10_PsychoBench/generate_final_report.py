import json
import csv
import os
import re
from statistics import mean, stdev

# ==========================================
# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
QUESTIONNAIRES_FILE = os.path.join(BASE_DIR, "questionnaires.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "report", "final_all_results_pivot.csv")

# List of questionnaires to include in order
SELECTED_TESTS = [
    "BFI", "16P", "EPQ-R", "DTDD", "BSRI", "ECR-R", "EIS", 
    "Empathy", "GSE", "ICB", "LMS", "LOT-R", "WLEIS", "CABIN"
]

# List of models (Folder Name / Display Name)
# Ensure these folder names exist in paper_benchmarks/10_PsychoBench/benchmark/results/
MODELS = [
    "Llama-70B-FP16",
    "Llama-70B-4bit",
    "Llama-70B-AWQ",
    "Llama-70B-GPTQ-INT4",
    "Llama-70B-GPTQ-INT8",
    "Qwen-7B-Chat",
    "Qwen-7B-Chat-4bit",
    "Qwen-7B-Chat-8bit",
    "Qwen-7B-Chat-GGUF-Q4_K_M",
    "Qwen-7B-Int4",
]

# Map (model_name, test_name) -> (custom_folder_name, custom_file_name)
SPECIAL_PATHS = {
    ("Qwen-7B-Chat-4bit", "CABIN"): ("Qwen-7B-Chat-BNB4-CABIN", "Qwen-7B-Chat-BNB4-CABIN-CABIN.csv"),
    ("Qwen-7B-Chat-8bit", "CABIN"): ("Qwen-7B-Chat-BNB8-CABIN", "Qwen-7B-Chat-BNB8-CABIN-CABIN.csv"),
    ("Qwen-7B-Chat", "CABIN"): ("Qwen-7B-Chat-FP16-CABIN", "Qwen-7B-Chat-FP16-CABIN-CABIN.csv"),
}

# ==========================================
# Core Functions
# ==========================================

def load_questionnaires():
    """Load questionnaire definitions from JSON."""
    with open(QUESTIONNAIRES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_csv_results(file_path, questionnaire):
    """
    Parses a single result CSV file.
    Returns list of dicts: [ {q_id: score, ...}, ... ] representing each run.
    """
    if not os.path.exists(file_path):
        # Only print warning if it's not the known missing file
        if "Qwen-7B-Chat-4bit-CABIN" not in file_path:
             # print(f"Warning: File not found: {file_path}")
             pass
        return None
        
    # DEBUG: Check if we are processing the special files
    is_debug = "BNB4" in file_path or "BNB8" in file_path
    # if is_debug:
    #     print(f"DEBUGGING Parse: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        # if is_debug: print("  > Empty rows")
        return None
    
    header = rows[0]
    # if is_debug: print(f"  > Header (first 5): {header[:5]}")
    
    # identify 'order' columns which mark the start of a block
    order_indices = [i for i, col in enumerate(header) if col.startswith("order")]
    
    # if is_debug: print(f"  > Order Indices: {order_indices}")
    
    # If no order columns found, try legacy format or return None
    if not order_indices:
        return None

    all_runs_data = [] # List of {q_idx: value}

    # Iterate over blocks created by 'order' columns
    for i in range(len(order_indices)):
        order_col_idx = order_indices[i]
        
        # Define the range of columns belonging to this block
        # Start: Immediately after order column
        start_col = order_col_idx + 1
        
        # End: Before next order column (and potentially skipping prompt column before it)
        if i + 1 < len(order_indices):
            next_order = order_indices[i+1]
            end_col = next_order - 1 # Assuming prompt column precedes order column
        else:
            end_col = len(header)
            
        # if is_debug: print(f"  > Block {i}: Start={start_col}, End={end_col}")

        # Verify valid data columns (heuristic: usually contains 'shuffle' or 'test')
        # But some files might be simpler. We'll iterate all in range.
        
        # Extract data for each column in this block
        for col_idx in range(start_col, end_col):
            # Skip if out of bounds
            if col_idx >= len(header): continue
            
            # Skip non-data columns if identifiable (e.g. Prompt columns)
            col_name = header[col_idx]
            if "Prompt" in col_name: continue

            # Extract run data
            run_data = {}
            for row_idx in range(1, len(rows)):
                row = rows[row_idx]
                if len(row) <= order_col_idx: continue
                
                # Get Question ID from Order Column
                # Value can be "1" or "1. Question Text"
                order_val = row[order_col_idx]
                q_idx = -1
                match = re.search(r'^(\d+)', str(order_val).strip())
                if match:
                    q_idx = int(match.group(1))
                else:
                    continue # Not a valid question row

                # Get Answer
                if col_idx < len(row):
                    ans_str = row[col_idx]
                    ans_match = re.search(r'\d+', str(ans_str))
                    if ans_match:
                        raw_val = int(ans_match.group())
                        
                        # Apply Reverse Scoring
                        if q_idx in questionnaire.get("reverse", []):
                            # Usually Scale + 1 - Val, or scale - val if scale is max?
                            # Based on previous scripts: `questionnaire["scale"] - val`
                            # NOTE: BFI in JSON has scale: 6. 1-6? Or 1-5?
                            # Standard BFI is 1-5. If JSON says 6, then 6-val flips 1->5, 5->1.
                            raw_val = questionnaire["scale"] - raw_val
                        
                        # Outlier Filtering
                        # Especially for CABIN which had model hallucination issues (outputting 7,8,9 on 1-5 scale)
                        if questionnaire["name"] == "CABIN":
                            if not (1 <= raw_val <= 5):
                                continue 
                        
                        run_data[q_idx] = raw_val
            
            # if is_debug and run_data:
                 # print(f"      > Col {col_idx}: Parsed {len(run_data)} questions. First 5 keys: {list(run_data.keys())[:5]}")

            if run_data:
                all_runs_data.append(run_data)
                
    return all_runs_data

def calculate_category_stats(runs_data, questionnaire):
    """
    Aggregates run data into Mean ± SD stats per category.
    Returns dict: {category_name: "Mean ± SD"}
    """
    if not questionnaire.get("categories"):
        return {}

    stats_results = {}
    
    for cat in questionnaire["categories"]:
        cat_name = cat["cat_name"]
        cat_questions = cat["cat_questions"]
        
        # Compute score for this category across all runs
        run_scores = []
        for run in runs_data:
            vals = [run[q] for q in cat_questions if q in run]
            
            if vals:
                if questionnaire.get("compute_mode") == "SUM":
                    run_scores.append(sum(vals))
                else: # AVG
                    run_scores.append(mean(vals))
        
        if run_scores:
            m = mean(run_scores)
            s = stdev(run_scores) if len(run_scores) > 1 else 0.0
            stats_results[cat_name] = f"{m:.2f} ± {s:.2f}"
            if questionnaire["name"] == "CABIN" and "BNB4" in str(runs_data): # Hacky check for debug
                 # This check won't work well because runs_data is list of dicts.
                 pass
        else:
            stats_results[cat_name] = "N/A"
            if questionnaire["name"] == "CABIN": 
                # print(f"DEBUG: N/A for {cat_name}. run_scores empty. runs_data len: {len(runs_data)}")
                pass
            
    return stats_results

def main():
    print("Step 1: Loading Questionnaires...")
    q_data = load_questionnaires()
    q_map = {q["name"]: q for q in q_data}
    
    # Prepare Pivot Headers
    # Row 1: Test Names
    # Row 2: Category Names
    header_test_names = [''] # First col empty (over Model Name)
    header_cat_names = ['Model']
    
    # Store mapping of (Test, Cat) for data lookup order
    column_mapping = [] # List of tuples (Test, Cat)

    for test_name in SELECTED_TESTS:
        if test_name not in q_map:
            print(f"Warning: {test_name} not found in questionnaires.json")
            continue
            
        q_config = q_map[test_name]
        if "categories" in q_config:
            for cat in q_config["categories"]:
                cat_name = cat["cat_name"]
                header_test_names.append(test_name)
                header_cat_names.append(cat_name)
                column_mapping.append((test_name, cat_name))

    # Initialize Rows for CSV
    # Structure: [ [ModelName, val1, val2...], ... ]
    csv_rows = []

    print("Step 2: Processing Models...")
    for model_name in MODELS:
        print(f"  > {model_name}")
        row = [model_name]
        
        # Pre-load all stats for this model
        model_stats = {} # (Test, Cat) -> Value string
        
        model_path = os.path.join(RESULTS_DIR, model_name)
        
        for test_name in SELECTED_TESTS:
            if test_name not in q_map: continue
            
            q_config = q_map[test_name]
            
            # Construct CSV Path
            is_debug = False
            if (model_name, test_name) in SPECIAL_PATHS:
                folder, filename = SPECIAL_PATHS[(model_name, test_name)]
                csv_full_path = os.path.join(RESULTS_DIR, folder, filename)
                is_debug = True
            else:
                # Standard naming convention
                candidates = [
                    f"{model_name}-{test_name}.csv",
                    f"{model_name}-Optimized-{test_name}.csv"
                ]
                csv_full_path = None
                for c in candidates:
                    cand_path = os.path.join(model_path, c)
                    if os.path.exists(cand_path):
                        csv_full_path = cand_path
                        break
                
                # Fallback if neither found (keep original logic to allow None return later)
                if not csv_full_path:
                    csv_full_path = os.path.join(model_path, f"{model_name}-{test_name}.csv")

            runs = parse_csv_results(csv_full_path, q_config)
            
            if runs:
                cat_results = calculate_category_stats(runs, q_config)
                # if is_debug:
                #    print(f"    > Calculated Stats for {test_name}: {list(cat_results.items())[:3]}") # Print first 3
                for cat, val_str in cat_results.items():
                    model_stats[(test_name, cat)] = val_str
            else:
                pass # Can't fill stats
        
        # Assemble Row
        for (t_name, c_name) in column_mapping:
            val = model_stats.get((t_name, c_name), "N/A")
            row.append(val)
            
        csv_rows.append(row)

    print(f"Step 3: Writing to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_test_names)
        writer.writerow(header_cat_names)
        writer.writerows(csv_rows)

    print("Done!")

if __name__ == "__main__":
    main()
