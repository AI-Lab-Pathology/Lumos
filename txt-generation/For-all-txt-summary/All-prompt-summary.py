
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, Iterable

from openai import OpenAI
from tqdm import tqdm


def read_text_with_fallback(path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "big5", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        f"Failed to decode {path} with tried encodings: {encodings}. Last error: {last_err}"
    )


API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not API_KEY:
    try:
        api_key_txt = os.path.join(os.getcwd(), "apikey.txt")
        if os.path.exists(api_key_txt):
            API_KEY = read_text_with_fallback(api_key_txt).strip()
    except Exception:
        API_KEY = ""

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner").strip()


INPUT_ROOT = r"txt_improve"
OUTPUT_ROOT = r"txt_summary"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


CATEGORIES = {"ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"}


PROMPT_FILE = r"Deepseek-summary.txt"


FILE_SUFFIX_FILTER = "_opt.txt"


MAX_TOKENS = 800
MAX_RETRIES = 3
WORKERS = 4
SLEEP_SECONDS = 3
SAVE_REASONING = True


def make_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY (env) and apikey.txt")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ===================== 读取 prompts.txt（多类别） =====================
def load_category_prompts(path: str) -> Dict[str, str]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = read_text_with_fallback(path)
    lines = text.splitlines()

    blocks: Dict[str, str] = {}
    key = None
    buf = []

    header_re = re.compile(r"^([A-Z]{3})[:：]\s*$")
    for raw in lines:
        line = raw.rstrip("\n")
        m = header_re.match(line.strip())
        if m:
            if key is not None:
                blocks[key] = _clean_prompt_text("\n".join(buf).strip())
                buf = []
            key = m.group(1).upper()
            continue
        buf.append(line)

    if key is not None:
        blocks[key] = _clean_prompt_text("\n".join(buf).strip())

    blocks = {k: v for k, v in blocks.items() if k in CATEGORIES}
    if not blocks:
        raise ValueError(f"No valid category prompts found in {path}. Use headers like 'ADI:' or 'ADI：'.")
    return blocks

def _clean_prompt_text(s: str) -> str:

    cleaned = []
    for ln in s.splitlines():
        if ln.lstrip().startswith("#"):
            ln = re.sub(r"^\s*#\s?", "", ln)
        cleaned.append(ln.rstrip())
    return "\n".join(cleaned).strip()


def build_summary_prompt(raw_text: str, category: str, prompt_map: Dict[str, str]) -> str:

    cat = category.upper()
    if cat not in prompt_map:
        raise KeyError(f"No prompt found for category '{cat}'. Available: {sorted(prompt_map.keys())}")

    template = prompt_map[cat].strip()
    return f"{template}\n\nOriginal:\n{raw_text}\n\nCondensed Summary:\n"


def call_model(client: OpenAI, final_prompt: str, max_retries: int = MAX_RETRIES) -> Tuple[str, Optional[str]]:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=MAX_TOKENS,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip() if SAVE_REASONING else None
            return content, reasoning
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_SECONDS)
    raise RuntimeError(f"⛔ Failed after {max_retries} retries: {last_err}")


def iter_files_with_category(root: str, categories: Iterable[str]) -> Iterable[Tuple[str, str]]:

    cats = {c.upper() for c in categories}
    for dirpath, _, filenames in os.walk(root):
        parts_upper = [p.upper() for p in dirpath.split(os.sep)]
        path_cats = [p for p in parts_upper if p in cats]
        guessed_cat = path_cats[-1] if path_cats else None

        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            if FILE_SUFFIX_FILTER and not fn.endswith(FILE_SUFFIX_FILTER):
                continue
            if guessed_cat is None:
                continue
            yield os.path.join(dirpath, fn), guessed_cat


def process_file(client: OpenAI,
                 file_path: str,
                 category: str,
                 prompt_map: Dict[str, str],
                 output_root: str) -> Optional[str]:

    try:
        raw = read_text_with_fallback(file_path).strip()
    except Exception as e:
        return f"⚠️ Read failed (skip): {file_path} -> {e}"
    if not raw:
        return f"⚠️ Empty (skip): {os.path.basename(file_path)}"


    out_dir = os.path.join(output_root, category.upper())
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_summary = os.path.join(out_dir, f"{base}_summary.txt")
    out_reason = os.path.join(out_dir, f"{base}_reasoning.txt")


    if os.path.exists(out_summary) and (not SAVE_REASONING or os.path.exists(out_reason)):
        return f"✅ Skipped (exists): {os.path.basename(file_path)}"


    try:
        final_prompt = build_summary_prompt(raw, category, prompt_map)
        summary, reasoning = call_model(client, final_prompt, MAX_RETRIES)

        with open(out_summary, "w", encoding="utf-8") as f_sum:
            f_sum.write(summary)
        if SAVE_REASONING:
            with open(out_reason, "w", encoding="utf-8") as f_r:
                f_r.write(reasoning or "No reasoning returned.")
        return f"✔ Processed: {os.path.basename(file_path)} [{category}]"
    except Exception as e:
        return f"❌ Failed: {os.path.basename(file_path)} [{category}] -> {e}"


def main():
    client = make_client()
    prompt_map = load_category_prompts(PROMPT_FILE)

    tasks = list(iter_files_with_category(INPUT_ROOT, CATEGORIES))
    if not tasks:
        print("⚠️ No files found or no category recognized in paths.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(process_file, client, fpath, cat, prompt_map, OUTPUT_ROOT): (fpath, cat)
            for (fpath, cat) in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            results.append(fut.result())

    for r in results:
        print(r)

if __name__ == "__main__":
    main()
