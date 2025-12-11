import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import subprocess
import gc

# Add paths for pyscent (adjust if needed)
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

# Model path from tree
model_path = 'Magicoder-6.7B-code-change-summ-impl'

# Prompt for claims
CLAIM_PROMPT_TEMPLATE = """
Summarize the code change in the following diff, including low-level modifications and high-level implications or risks:
{diff}
Generate claims as a bullet list.
"""

# Prompt for refinement (using same LLM for refinement)
REFINE_PROMPT_TEMPLATE = """
The original human review is: {review}
The CRScore feedback is: comprehensiveness={comp}, relevance={rel}, conciseness={con}. Identified gaps: {feedback}
Refine the review to address deficiencies, incorporate missing information, remove irrelevant content, improve clarity, while preserving original intent.
Output the enhanced review.
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
            pmd_bin = 'pmd-bin-7.6.0/bin/pmd'  # Assume in current dir or PATH
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

def refine_review(model, tokenizer, review, scores, feedback, device):
    prompt = REFINE_PROMPT_TEMPLATE.format(review=review, comp=scores['comprehensiveness'], rel=scores['relevance'], con=scores['conciseness'], feedback=feedback)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    refined = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return refined

# Load model and embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) if device == 'cuda' else None

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto" if device == 'cuda' else None,
    dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
    local_files_only=True
)

embedder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu')

# Example input (hardcode or load from dataset; adjust)
code_diff = """
- old_line = 1
+ new_line = 2
"""  # Replace with real diff
human_review = "This change fixes a bug."  # Replace with real review
language = 'python'  # or 'java', 'javascript'

# Parameters
threshold = 0.5  # Overall score threshold (average of three)
max_iterations = 5
feedback = "Improve comprehensiveness by covering more claims and smells."  # From CRScore gaps

# Iterative refinement
current_review = human_review
iteration = 0

while iteration < max_iterations:
    iteration += 1
    print(f"Iteration {iteration}: Evaluating review...")
    
    old_code = '\n'.join([line[1:] for line in code_diff.split('\n') if line.startswith('-')])
    new_code = '\n'.join([line[1:] for line in code_diff.split('\n') if line.startswith('+')])
    
    claims = generate_claims(model, tokenizer, code_diff, device)
    smells = detect_code_smells('temp', language, old_code, new_code)
    pseudo_refs = claims + smells
    
    scores = compute_crscore(pseudo_refs, current_review, embedder)
    overall_score = (scores['conciseness'] + scores['comprehensiveness'] + scores['relevance']) / 3
    
    print(f"Scores: {scores}, Overall: {overall_score}")
    
    if overall_score >= threshold:
        print("Review meets threshold. Final review:")
        print(current_review)
        break
    
    # Refine
    current_review = refine_review(model, tokenizer, current_review, scores, feedback, device)
    print("Refined review:")
    print(current_review)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if iteration == max_iterations:
    print("Max iterations reached. Final review:")
    print(current_review)