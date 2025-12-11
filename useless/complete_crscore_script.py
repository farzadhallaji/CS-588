# complete_crscore_script.py
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import os
import subprocess
import sys
import gc
from tqdm import tqdm

# Add paths for pyscent
sys.path.append('CRScore/src/code_smells/pyscent/src')
sys.path.append('CRScore/src/code_smells/pyscent/src/Detector')
sys.path.append('CRScore/src/code_smells/pyscent/tools')

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
    # Slice to new tokens only
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # Parse bullets
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
            # Direct PMD with multiple rulesets
            pmd_bin = 'pmd-bin-7.6.0/bin/pmd'
            rules = 'category/java/design.xml,category/java/bestpractices.xml,category/java/errorprone.xml'
            smells_old = []
            smells_new = []
            for file, smells_list in [(old_file, smells_old), (new_file, smells_new)]:
                result = subprocess.run([pmd_bin, 'check', '-d', file, '-R', rules, '-f', 'text'], capture_output=True, text=True)
                if result.returncode in [0, 4]:  # Violations found or not
                    smells_list.extend([line for line in result.stdout.splitlines() if 'Violation' in line or 'Problem' in line])
                else:
                    print(f"PMD failed for {file}: {result.stderr}")
            smells = list(set(smells_new) - set(smells_old))
        elif language == 'javascript':
            # Try repo script
            result = subprocess.run(['python', 'CRScore/scripts/detect_javascript_code_smells.py', '--old', old_file, '--new', new_file], capture_output=True, text=True)
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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if device == 'cuda' else None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto" if device == 'cuda' else None,
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    )
    
    embedder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu')
    
    with open(args.dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f][:args.max_items]
    
    results = []
    done_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            results = json.load(f)
        done_ids = {r['id'] for r in results}
        print(f"Resuming from {len(results)} completed items.")
    
    try:
        for idx, item in enumerate(tqdm(dataset)):
            item_id = item.get('id', idx)
            if item_id in done_ids:
                continue
            
            diff = item.get('diff', item.get('code', ''))  # Use 'code' if 'diff' empty
            review = item.get('msg', item.get('review', ''))
            language = item.get('lang', 'java').lower()
            
            old_code = '\n'.join([line[1:] for line in diff.split('\n') if line.startswith('-')]) if diff else ''
            new_code = '\n'.join([line[1:] for line in diff.split('\n') if line.startswith('+')]) if diff else diff  # Use code as new if no diff
            
            claims = generate_claims(model, tokenizer, diff, device)
            
            smells = detect_code_smells(item_id, language, old_code, new_code)
            
            pseudo_refs = claims + smells
            
            scores = compute_crscore(pseudo_refs, review, embedder, args.threshold)
            
            results.append({
                'id': item_id,
                'diff': diff,
                'review': review,
                'claims': claims,
                'smells': smells,
                'scores': scores
            })
            
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Interrupted! Saving partial results...")
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CRScore on a dataset")
    parser.add_argument('--model_path', type=str, default='Magicoder-6.7B-code-change-summ-impl')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='crscore_results.json')
    parser.add_argument('--threshold', type=float, default=0.7314)
    # parser.add_argument('--max_items', type=int, default=10169, help='Max items to process (for testing)')
    parser.add_argument('--max_items', type=int, default=10, help='Max items to process (for testing)')
    args = parser.parse_args()
    main(args)

# python complete_crscore_script.py --dataset_path CRScore/data/Comment_Generation/msg-test.jsonl