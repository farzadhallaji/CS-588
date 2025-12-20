# %%
import torch, time
print("cuda?", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")


# %%
import torch

import sys, os
sys.path.append(os.path.join(os.getcwd(), "CRScore"))
from sentence_transformers import SentenceTransformer, util
from src.metrics.claim_based.relevance_score import split_claims_and_impl

claims_text = """
1. The code change is in the ProtocGapicPluginGeneratorTest class.
2. The parameter is now "language=java,transport=grpc".
3. Previously it was "language=java" only.
Implications:
1. Codegen now targets Java over gRPC.
"""
claims = split_claims_and_impl(claims_text)
review = "Adds gRPC transport to the Java codegen; check clients still build."

device = "cuda" if torch.cuda.is_available() else "cpu"

sbert = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)

change_enc = sbert.encode(claims, convert_to_tensor=True, device=device, show_progress_bar=False)
review_sentences = [s for s in review.split(".") if s.strip()] or [review]
review_enc = sbert.encode(review_sentences, convert_to_tensor=True, device=device, show_progress_bar=False)

sts = util.cos_sim(change_enc, review_enc)

tau = 0.7314

prec_mask = (sts.max(dim=0).values > tau).float()
P = prec_mask.mean().item()
rec_mask = (sts > tau).sum(dim=1) > 0
R = rec_mask.float().mean().item()
F = 0 if (P + R) == 0 else 2 * P * R / (P + R)

print(f"Con={P:.3f}  Comp={R:.3f}  Rel={F:.3f}")


# %%
import json
from pathlib import Path
from itertools import islice

ROOT = Path("/home/ri/Desktop/S2/DS")  # repo root
split = "test"                         # msg-test.jsonl
idx = 6                                # pick any index that exists

data_path = ROOT / "package" / "Comment_Generation" / f"msg-{split}.jsonl"
rec = json.loads(next(islice(data_path.open(), idx, None)))
print(f"idx={idx}, lang={rec['lang']}, msg={rec['msg']}")

# map language -> smell folder/extension
smell_dir = {"py": "python_code_smells", "java": "java_code_smells", "js": "javascript_code_smells"}
smell_ext = {"py": "json", "java": "txt", "js": "txt"}
smell_path = ROOT / "package" / "CRScore" / "experiments" / smell_dir[rec["lang"]] / f"test{idx}.{smell_ext[rec['lang']]}"

if smell_path.suffix == ".json":
    smell = json.load(smell_path.open())
else:
    smell = smell_path.read_text()

print("smell path:", smell_path)
print("smell data:", smell)


# %%
# Assumes you run this from /home/ri/Desktop/S2/DS/package
import sys, json
from pathlib import Path
from itertools import islice

# Make CRScore importable
ROOT = Path.cwd() / "CRScore"
sys.path.append(str(ROOT))

from src.datautils import read_jsonl
from src.metrics.claim_based.relevance_score import (
    RelevanceScorer,
    split_claims_and_impl,
    process_python_smells,
    process_java_smells,
    process_javascript_smells,
    filter_by_changed_lines,
)
from scripts.create_code_smell_analysis_data import generate_newf

# ------- choose the instance -------
idx = 6          # 0-based row in msg-test.jsonl; change as needed
split = "test"   # uses msg-test.jsonl + test_set_codepatch_ranges.json

# ------- load the review + patch -------
data = read_jsonl(ROOT / "data" / "Comment_Generation" / f"msg-{split}.jsonl")
rec = data[idx]
review = rec["msg"]
diff = rec["patch"]
lang = rec["lang"]

# ------- load code-change claims (LLM summary with implications) -------
claims_file = ROOT / "experiments" / "code_change_summ_finetune_impl" / "Magicoder-S-DS-6.7B.jsonl"
claims_text = json.loads(next(islice(claims_file.open(), idx, None)))["response"]
claims = split_claims_and_impl(claims_text)

# ------- load smells (language-specific) and filter to changed lines -------
patch_ranges = json.load(open(ROOT / "data" / "Comment_Generation" / "test_set_codepatch_ranges.json"))
new_file, _ = generate_newf(rec["oldf"], rec["patch"])
range_for_idx = patch_ranges[f"test{idx}"]

smell_dir = {"py": "python_code_smells", "java": "java_code_smells", "js": "javascript_code_smells"}
smell_ext = {"py": "json", "java": "txt", "js": "txt"}
smell_path = ROOT / "experiments" / smell_dir[lang] / f"test{idx}.{smell_ext[lang]}"

smells_raw = []
if smell_path.exists():
    if lang == "py":
        smells_raw = process_python_smells(smell_path, smell_path.name)
    elif lang == "java":
        smells_raw = process_java_smells(smell_path, smell_path.name)
    else:
        smells_raw = process_javascript_smells(smell_path, smell_path.name)

smells = filter_by_changed_lines(smells_raw, range_for_idx, new_file, diff)

# ------- score review relevance (precision/recall/F1 over claims+smells) -------
scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=0.85)
P, R, sts_matrix = scorer.compute_inst(claims + smells, review, debug=True)  # debug prints claim alignment
F1 = 0 if (P + R) == 0 else (2 * P * R) / (P + R)

print(f"idx={idx}, lang={lang}")
print(f"P={P:.3f}, R={R:.3f}, F1={F1:.3f}")
print(f"{len(claims)} claims + {len(smells)} smells used")


# %%
# Run from /home/ri/Desktop/S2/DS/package
import sys, json
from pathlib import Path

ROOT = Path.cwd() / "CRScore"
sys.path.append(str(ROOT))

from src.metrics.claim_based.relevance_score import (
    RelevanceScorer,
    split_claims_and_impl,
    filter_by_changed_lines,
    process_python_smells,
    process_java_smells,
    process_javascript_smells,
)
from scripts.create_code_smell_analysis_data import generate_newf

# ----------- YOUR INPUTS -----------
review_text = """<put the review you want to score here>"""

claims_text = """<put the code-change summary/implications text here; could be your own or model output>"""
claims = split_claims_and_impl(claims_text)

# If you have a unified diff + original file, fill these; else leave empty and smells will be skipped.
patch = r"""<paste unified diff with @@ hunk header here>"""
old_file_contents = """<paste full original file contents here>"""

# Optional: precomputed smell file path (py/json, java/txt, js/txt), else leave None and provide manual smells below.
smell_file = None  # e.g., ROOT / "experiments/python_code_smells/test6.json"

# Optional: manual smells if you want to supply them directly (list of (text, line_no))
manual_smells = []  # e.g., [("line 42, Long Method smell ...", 42)]
# -----------------------------------

# Build new file from diff if provided
new_file = ""
if patch.strip() and old_file_contents.strip():
    new_file, _ = generate_newf(old_file_contents, patch)

# Load smells if a file is provided
smells_raw = []
if smell_file:
    if smell_file.suffix == ".json":
        smells_raw = process_python_smells(smell_file, smell_file.name)
    elif smell_file.suffix == ".txt":
        # crude language guess from parent folder name
        if "java" in smell_file.parts[-2]:
            smells_raw = process_java_smells(smell_file, smell_file.name)
        else:
            smells_raw = process_javascript_smells(smell_file, smell_file.name)

# Combine with any manual smells you typed in
smells_raw.extend(manual_smells)

# Filter smells to changed lines if we have diff + new file; otherwise keep as-is
smells = smells_raw
if new_file and patch:
    smells = filter_by_changed_lines(smells_raw, [1, 10**9], new_file, patch)  # wide range; uses actual changed lines internally

# Assemble claim/smell list and score
claims_and_smells = claims + [s[0] if isinstance(s, (list, tuple)) else s for s in smells]

scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=0.85)
P, R, sts = scorer.compute_inst(claims_and_smells, review_text, debug=True)
F1 = 0 if (P + R) == 0 else 2 * P * R / (P + R)

print(f"P={P:.3f}, R={R:.3f}, F1={F1:.3f}")
print(f"{len(claims)} claims + {len(smells)} smells used")


# %%
# Run from /home/ri/Desktop/S2/DS/package
import sys, json
from pathlib import Path

ROOT = Path.cwd() / "CRScore"
sys.path.append(str(ROOT))

from src.metrics.claim_based.relevance_score import (
    RelevanceScorer,
    split_claims_and_impl,
    filter_by_changed_lines,
    process_python_smells,
    process_java_smells,
    process_javascript_smells,
)
from scripts.create_code_smell_analysis_data import generate_newf

# ----------- YOUR INPUTS -----------
review_text = """<put the review you want to score here>"""

claims_text = """<put the code-change summary/implications text here; could be your own or model output>"""
claims = split_claims_and_impl(claims_text)

# If you have a unified diff + original file, fill these; else leave empty and smells will be skipped.
patch = r"""<paste unified diff with @@ hunk header here>"""
old_file_contents = """<paste full original file contents here>"""

# Optional: precomputed smell file path (py/json, java/txt, js/txt), else leave None and provide manual smells below.
smell_file = None  # e.g., ROOT / "experiments/python_code_smells/test6.json"

# Optional: manual smells if you want to supply them directly (list of (text, line_no))
manual_smells = []  # e.g., [("line 42, Long Method smell ...", 42)]
# -----------------------------------

# Build new file from diff if provided
new_file = ""
if patch.strip() and old_file_contents.strip():
    new_file, _ = generate_newf(old_file_contents, patch)

# Load smells if a file is provided
smells_raw = []
if smell_file:
    if smell_file.suffix == ".json":
        smells_raw = process_python_smells(smell_file, smell_file.name)
    elif smell_file.suffix == ".txt":
        # crude language guess from parent folder name
        if "java" in smell_file.parts[-2]:
            smells_raw = process_java_smells(smell_file, smell_file.name)
        else:
            smells_raw = process_javascript_smells(smell_file, smell_file.name)

# Combine with any manual smells you typed in
smells_raw.extend(manual_smells)

# Filter smells to changed lines if we have diff + new file; otherwise keep as-is
smells = smells_raw
if new_file and patch:
    smells = filter_by_changed_lines(smells_raw, [1, 10**9], new_file, patch)  # wide range; uses actual changed lines internally

# Assemble claim/smell list and score
claims_and_smells = claims + [s[0] if isinstance(s, (list, tuple)) else s for s in smells]

scorer = RelevanceScorer(model_path="mixedbread-ai/mxbai-embed-large-v1", hi_sim_thresh=0.85)
P, R, sts = scorer.compute_inst(claims_and_smells, review_text, debug=True)
F1 = 0 if (P + R) == 0 else 2 * P * R / (P + R)

print(f"P={P:.3f}, R={R:.3f}, F1={F1:.3f}")
print(f"{len(claims)} claims + {len(smells)} smells used")


# %%


# %%
import argparse
import json
import textwrap
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_records(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def extract_review(entry: Dict[str, Any]) -> str:
    for key in ("review", "pred_review", "pred", "msg", "response"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_diff(entry: Dict[str, Any]) -> str:
    for key in ("diff", "code_change", "patch"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def compute_score(entry: Dict[str, Any], mode: str) -> Optional[float]:

    scores: Dict[str, Any] = entry.get("scores") or entry.get("metric_scores") or {}
    has_scores_dict = isinstance(scores, dict)

    if mode in ("auto", "score"):
        val = entry.get("score")
        if isinstance(val, (int, float)):
            return float(val)
        if mode == "score":
            return None

    if not has_scores_dict:
        return None

    if "relevance" in scores:
        return float(scores["relevance"])
    if "F" in scores:
        return float(scores["F"])

    p = scores.get("conciseness") if "conciseness" in scores else scores.get("P")
    r = scores.get("comprehensiveness") if "comprehensiveness" in scores else scores.get("R")
    if isinstance(p, (int, float)) and isinstance(r, (int, float)):
        denom = p + r
        return 0.0 if denom == 0 else (2 * p * r) / denom

    numeric_scores = [v for v in scores.values() if isinstance(v, (int, float))]
    return mean(numeric_scores) if numeric_scores else None


def select_extremes(
    records: Iterable[Dict[str, Any]], k: int, mode: str
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    scored = []
    for rec in records:
        score = compute_score(rec, mode)
        if score is not None:
            scored.append((rec, score))
    if not scored:
        return [], []
    ordered = sorted(scored, key=lambda item: item[1])
    worst = ordered[:k]
    best = list(reversed(ordered[-k:]))
    return worst, best


# %%


# %%


# %%


# %%


# %%
import json
from pathlib import Path
from itertools import islice

def get_lowest_f_review(
    scores_path=Path("CRScore/all_model_rel_scores_thresh_0.7314.json"),
    data_path=Path("Comment_Generation/msg-test.jsonl"),
    system="ground_truth",
):
    scores = json.loads(scores_path.read_text())
    f_vals = scores[system]["F"]
    min_idx = min(range(len(f_vals)), key=f_vals.__getitem__)
    rec = json.loads(next(islice(data_path.open(), min_idx, None)))
    return {
        "idx": min_idx,
        "id": rec.get("id"),
        "lang": rec.get("lang"),
        "review": rec.get("msg"),
        "patch": rec.get("patch"),
    }

bad = get_lowest_f_review()
print(bad.keys())
print("idx:", bad["idx"], "id:", bad["id"], "lang:", bad["lang"])
print("review:", bad["review"])
print("patch:\n", bad["patch"])


# %%
import json
from pathlib import Path
from itertools import islice

def get_lowest_f_review_with_score(
    scores_path=Path("CRScore/all_model_rel_scores_thresh_0.7314.json"),
    data_path=Path("Comment_Generation/msg-test.jsonl"),
    system="ground_truth",
):
    scores = json.loads(scores_path.read_text())
    f_vals = scores[system]["F"]
    p_vals = scores[system]["P"]
    r_vals = scores[system]["R"]

    min_idx = min(range(len(f_vals)), key=f_vals.__getitem__)
    rec = json.loads(next(islice(data_path.open(), min_idx, None)))
    return {
        "idx": min_idx,
        "id": rec.get("id"),
        "lang": rec.get("lang"),
        "review": rec.get("msg"),
        "patch": rec.get("patch"),
        "P": p_vals[min_idx],
        "R": r_vals[min_idx],
        "F": f_vals[min_idx],
    }
bad = get_lowest_f_review_with_score()
print(f"idx={bad['idx']} id={bad['id']} lang={bad['lang']}"); print(f"P={bad['P']:.3f} R={bad['R']:.3f} F={bad['F']:.3f}"); print("review:", bad["review"]); print("patch:\n", bad["patch"])

# %%
import json
from itertools import islice
from pathlib import Path
from typing import List, Optional

ROOT = Path("/home/ri/Desktop/S2/DS/package")
import sys
sys.path.append(str(ROOT / "CRScore"))
from src.metrics.claim_based.relevance_score import split_claims_and_impl

def get_java_claims(
    split: str = "test",
    claims_file: Path = ROOT / "CRScore/experiments/code_change_summ_finetune_impl/Magicoder-S-DS-6.7B.jsonl",
    idx: Optional[int] = None,
) -> List[str]:
    """
    Return the claims list for a Java example.
    - split: which msg-{split}.jsonl to read (train/dev/test)
    - claims_file: JSONL with code-change summaries+implications (aligned by index with msg file)
    - idx: specific index; if None, pick the first Java example in msg-{split}.jsonl
    """
    msg_path = ROOT / "Comment_Generation" / f"msg-{split}.jsonl"
    if idx is None:
        with msg_path.open() as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                if rec.get("lang") == "java":
                    idx = i
                    break
        if idx is None:
            raise ValueError(f"No Java examples found in {msg_path}")
    claims_text = json.loads(next(islice(claims_file.open(), idx, None)))["response"]
    claims = split_claims_and_impl(claims_text)
    return claims

claims = get_java_claims(split="test")
print("Num claims:", len(claims))
for c in claims:
    print("-", c)


# %%
import requests
from textwrap import dedent

SYSTEM = dedent("""\
You are a senior code reviewer. Improve the review so it is concise, specific, and aligned to the code diff and claims.
- Address the main effects of the diff.
- Call out correctness, safety, testing, and edge cases that matter.
- Avoid generic or unrelated style nits.
Output only the refined review text.
""")

def refine_with_ollama(review, claims, diff, model="llama3:8b-instruct-q4_0"):
    user = f"Current review:\n{review}\n\nClaims:\n{claims}\n\nDiff:\n{diff}"
    resp = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "llama3:8b-instruct-q4_0",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.2},
            "stream": False, 
        },
        timeout=60,
    )
    resp.raise_for_status()
    refined = resp.json()["message"]["content"].strip()

    return refined

review = "can we also test for `transport=rest`?"
claims = """- ProtocGapicPluginGeneratorTest now sets language=java,transport=grpc.
- Previously it set language=java only.
- This enables grpc transport for Java codegen."""
diff = """@@ -53,7 +53,7 @@ public class ProtocGapicPluginGeneratorTest {
-            .setParameter("language=java")
+            .setParameter("language=java,transport=grpc")
 }"""

refined = refine_with_ollama(review, claims, diff)
print(refined)


# %%
import sys
from pathlib import Path
from typing import Tuple, Union, Iterable
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("/home/ri/Desktop/S2/DS/package/CRScore")
sys.path.append(str(ROOT))
from src.metrics.claim_based.relevance_score import RelevanceScorer, split_claims_and_impl

def crscore(
    review_text: str,
    claims_input: Union[str, Iterable[str]],
    tau: float = 0.7314,
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
    use_gpu: bool = False,  # default CPU to avoid OOM
) -> Tuple[float, float, float, str]:
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    scorer = RelevanceScorer(model_path=model_path, hi_sim_thresh=tau)
    scorer.sbert = SentenceTransformer(model_path, device=device)
    claims = split_claims_and_impl(claims_input) if isinstance(claims_input, str) else [str(c) for c in claims_input]
    P, R, _ = scorer.compute_inst(claims, review_text, debug=False)
    F = 0.0 if (P + R) == 0 else (2 * P * R) / (P + R)
    return P, R, F, device

P, R, F, device = crscore(refined, claims, tau=0.7314)
print(f"P={P:.3f}, R={R:.3f}, F={F:.3f}")
print("Refined review:\n", refined)
print(f"CRScore @ tau=0.7314 -> P={P:.3f} R={R:.3f} F={F:.3f}")


# %%
import sys
from pathlib import Path
from typing import Tuple, Union, Iterable
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("/home/ri/Desktop/S2/DS/package/CRScore")
sys.path.append(str(ROOT))
from src.metrics.claim_based.relevance_score import RelevanceScorer, split_claims_and_impl

def crscore(
    review_text: str,
    claims_input: Union[str, Iterable[str]],
    tau: float = 0.7314,
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
    use_gpu: bool = False,  # default CPU to avoid OOM
) -> Tuple[float, float, float, str]:
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    scorer = RelevanceScorer(model_path=model_path, hi_sim_thresh=tau)
    scorer.sbert = SentenceTransformer(model_path, device=device)
    claims = split_claims_and_impl(claims_input) if isinstance(claims_input, str) else [str(c) for c in claims_input]
    P, R, _ = scorer.compute_inst(claims, review_text, debug=False)
    F = 0.0 if (P + R) == 0 else (2 * P * R) / (P + R)
    return P, R, F, device




refined = """The change correctly adds `transport=grpc` to the test parameters, enabling gRPC transport for Java codegen as claimed, without altering prior language=java behavior. For completeness, add a test case for `transport=rest` to verify it doesn't regress or conflict with gRPC logic—focus on edge cases like mixed transports or invalid combos. No safety issues noted, but ensure the generator handles absent transport params gracefully."""

P, R, F, device = crscore(refined, claims, tau=0.7314)
print(f"P={P:.3f}, R={R:.3f}, F={F:.3f}")
print("Refined review:\n", refined)
print(f"CRScore @ tau=0.7314 -> P={P:.3f} R={R:.3f} F={F:.3f}")


# %%
import sys
from pathlib import Path
from typing import Tuple, Union, Iterable
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("/home/ri/Desktop/S2/DS/package/CRScore")
sys.path.append(str(ROOT))
from src.metrics.claim_based.relevance_score import RelevanceScorer, split_claims_and_impl

def crscore(
    review_text: str,
    claims_input: Union[str, Iterable[str]],
    tau: float = 0.7314,
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
    use_gpu: bool = False,  # default CPU to avoid OOM
) -> Tuple[float, float, float, str]:
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    scorer = RelevanceScorer(model_path=model_path, hi_sim_thresh=tau)
    scorer.sbert = SentenceTransformer(model_path, device=device)
    claims = split_claims_and_impl(claims_input) if isinstance(claims_input, str) else [str(c) for c in claims_input]
    P, R, _ = scorer.compute_inst(claims, review_text, debug=False)
    F = 0.0 if (P + R) == 0 else (2 * P * R) / (P + R)
    return P, R, F, device




refined = """The change correctly adds `transport=grpc` to the test parameters, enabling gRPC transport for Java codegen as claimed, without altering prior language=java behavior. For completeness, add a test case for `transport=rest` to verify it doesn't regress or conflict with gRPC logic—focus on edge cases like mixed transports or invalid combos. No safety issues noted, but ensure the generator handles absent transport params gracefully."""

P, R, F, device = crscore(refined, claims, tau=0.7314)
print(f"P={P:.3f}, R={R:.3f}, F={F:.3f}")
print("Refined review:\n", refined)
print(f"CRScore @ tau=0.7314 -> P={P:.3f} R={R:.3f} F={F:.3f}")


# %%


# %%


# %%



