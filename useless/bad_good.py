import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import subprocess
import sys
from tqdm import tqdm
import gc
import argparse

# Add paths for pyscent (relative from 'runnnnn')
sys.path.append('../package/CRScore/src/code_smells/pyscent/src')
sys.path.append('../package/CRScore/src/code_smells/pyscent/src/Detector')
sys.path.append('../package/CRScore/src/code_smells/pyscent/tools')

# Try pyscent import
detect_smells = None
try:
    from detector import detect_smells
    print("pyscent import successful")
except ImportError as e:
    print(f"Warning: pyscent import failed: {e}. Skipping Python smell detection.")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))

# Model path fixed (local dir from 'runnnnn')
model_path = '../package/Magicoder-6.7B-code-change-summ-impl'

# Dataset path fixed
dataset_path = '../package/Comment_Generation/msg-test.jsonl'  # Or '../package/Code_Refinement/ref-test.jsonl' for code diffs

# Prompt for claims
CLAIM_PROMPT_TEMPLATE = """
Summarize the code change in the following diff, including low-level modifications and high-level implications or risks. If no diff, summarize the code snippet:
{diff}
Generate claims as a bullet list.
"""

def generate_claims(model, tokenizer, diff_text, device):
    prompt = CLAIM_PROMPT_TEMPLATE.format(diff=diff_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    claims = []
    for line in generated.splitlines():
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('*') or (line[0].isdigit() and '.' in line)):
            claims.append(line.lstrip('-* ').lstrip('0123456789. ').strip())
    return [c for c in claims if c.lower() not in stop_words and c]

def detect_code_smells(diff_id, language, old_code, new_code):
    if not old_code and not new_code:
        return []
    smells = []
    old_file = f'temp_old_{diff_id}.{language}'
    new_file = f'temp_new_{diff_id}.{language}'
    with open(old_file, 'w') as f:
        f.write(old_code)
    with open(new_file, 'w') as f:
        f.write(new_code)
    
    try:
        if language == 'python' and detect_smells is not None:
            smells_old = detect_smells(old_file)
            smells_new = detect_smells(new_file)
            smells = list(set(smells_new) - set(smells_old))
        elif language == 'java':
            pmd_bin = '../package/pmd-bin-7.6.0/bin/pmd'  # Fixed relative path from 'runnnnn'
            rules = 'category/java/design.xml,category/java/bestpractices.xml,category/java/errorprone.xml'
            smells_old = []
            smells_new = []
            for file, smells_list in [(old_file, smells_old), (new_file, smells_new)]:
                result = subprocess.run([pmd_bin, 'check', '-d', file, '-R', rules, '-f', 'text'], capture_output=True, text=True)
                if result.returncode in [0, 4]:
                    smells_list.extend([line for line in result.stdout.splitlines() if 'Violation' in line or 'Problem' in line])
                else:
                    print(f"PMD failed for {file}: {result.stderr}")
            smells = list(set(smells_new) - set(smells_old))
        elif language == 'javascript':
            result = subprocess.run(['python', '../package/CRScore/scripts/detect_javascript_code_smells.py', '--old', old_file, '--new', new_file], capture_output=True, text=True)  # Fixed relative
            if result.returncode == 0:
                smells = result.stdout.splitlines()
            else:
                print(f"JS detection failed: {result.stderr}")
    except Exception as e:
        print(f"Smell detection error: {e}")
    finally:
        if os.path.exists(old_file):
            os.remove(old_file)
        if os.path.exists(new_file):
            os.remove(new_file)
    return smells

def compute_crscore(pseudo_refs, review_text, embedder, threshold=0.7314):
    review_sents = nltk.sent_tokenize(review_text)
    if not review_sents or not pseudo_refs:
        return {"conciseness": 0.0, "comprehensiveness": 0.0, "relevance": 0.0}
    
    p_embs = embedder.encode(pseudo_refs, convert_to_tensor=True)
    r_embs = embedder.encode(review_sents, convert_to_tensor=True)
    
    sim_matrix = util.cos_sim(r_embs, p_embs)
    
    con = (sim_matrix.max(dim=1)[0] > threshold).float().mean().item()
    comp = (sim_matrix.max(dim=0)[0] > threshold).float().mean().item()
    rel = (2 * con * comp) / (con + comp) if (con + comp) > 0 else 0.0
    
    return {"conciseness": con, "comprehensiveness": comp, "relevance": rel}

# Thresholds for categorization
GOOD_THRESH = 0.7
MID_THRESH = 0.3

# Output folder for categorized
output_dir = 'categorized_reviews'
os.makedirs(output_dir, exist_ok=True)

# Processed folder for raw entries (to check and resume)
processed_dir = 'processed'
os.makedirs(processed_dir, exist_ok=True)

# Load model and embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) if device == 'cuda' else None

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)  # Force local, no HF download
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto" if device == 'cuda' else None,
    dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
    local_files_only=True  # Force local, no HF download
)

embedder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu')

# Load dataset
data_points = []
with open(dataset_path, 'r') as f:
    for line in f:
        data_points.append(json.loads(line))

# Process: compute for each, save processed, categorize
for metric in ['conciseness', 'comprehensiveness', 'relevance']:
    metric_dir = os.path.join(output_dir, metric)
    os.makedirs(metric_dir, exist_ok=True)
    
    good, mid, bad = [], [], []
    
    for i, item in enumerate(tqdm(data_points)):
        processed_file = os.path.join(processed_dir, f'entry_{i}.json')
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                entry = json.load(f)
            score = entry['scores'][metric]
        else:
            diff = item.get('diff', '')
            review = item.get('msg', '')
            language = item.get('lang', 'java').lower()  # Default java
            
            old_code = '\n'.join([line[1:] for line in diff.split('\n') if line.startswith('-')])
            new_code = '\n'.join([line[1:] for line in diff.split('\n') if line.startswith('+')])
            
            claims = generate_claims(model, tokenizer, diff, device)
            smells = detect_code_smells(i, language, old_code, new_code)
            pseudo_refs = claims + smells
            
            scores = compute_crscore(pseudo_refs, review, embedder)
            
            entry = {
                'id': i,
                'diff': diff,
                'review': review,
                'lang': language,
                'claims': claims,
                'smells': smells,
                'scores': scores
            }
            
            with open(processed_file, 'w') as f:
                json.dump(entry, f, indent=4)
            
            score = scores[metric]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if score > GOOD_THRESH:
            good.append(entry)
        elif score > MID_THRESH:
            mid.append(entry)
        else:
            bad.append(entry)
    
    # Save categorized
    with open(os.path.join(metric_dir, 'good.json'), 'w') as f:
        json.dump(good, f, indent=4)
    with open(os.path.join(metric_dir, 'mid.json'), 'w') as f:
        json.dump(mid, f, indent=4)
    with open(os.path.join(metric_dir, 'bad.json'), 'w') as f:
        json.dump(bad, f, indent=4)

print(f"Categorized reviews saved to '{output_dir}' with subfolders for each metric. Processed entries saved to '{processed_dir}' for checking/resume. Use --max_items via argparse to limit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and categorize CRScore")
    parser.add_argument('--max_items', type=int, default=10, help='Max items to process')
    args = parser.parse_args()
    # In data_points load: data_points = data_points[:args.max_items]
    # Adjust the loop: for i, item in enumerate(tqdm(data_points[:args.max_items])):