import os
import subprocess
import logging
import time
from typing import Optional, List
from functools import wraps
from collections import deque
from diff_match_patch import diff_match_patch
from ollama import chat, ChatResponse 
import hashlib
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b")
PATCH_OUTPUT_FILE = "ai_patch_output.txt" 

# Add configurations
MAX_FILE_SIZE = 1024 * 1024  # 1MB
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
EXCLUDED_PATTERNS = ['.git', '__pycache__', 'venv', 'env', '.pytest_cache']
MAX_WORKERS = 4
SIMILARITY_THRESHOLD = 0.95
BACKUP_DIR = Path("code_backups")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def retry_on_error(max_attempts: int = RETRY_ATTEMPTS, delay: int = RETRY_DELAY):
    """Decorator to retry functions on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class ProcessingStats:
    def __init__(self):
        self.files_processed = 0
        self.files_modified = 0
        self.files_skipped = 0
        self.errors = 0

    def print_summary(self):
        logging.info("Processing Summary:")
        logging.info(f"Files Processed: {self.files_processed}")
        logging.info(f"Files Modified: {self.files_modified}")
        logging.info(f"Files Skipped: {self.files_skipped}")
        logging.info(f"Errors: {self.errors}")

def get_all_python_files():
    """Recursively finds all Python files in the repository, excluding specified patterns."""
    python_files = []
    for root, dirs, files in os.walk(os.getcwd()):
        # Remove excluded directories
        for excluded in EXCLUDED_PATTERNS:
            if excluded in dirs:
                dirs.remove(excluded)
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                # Check file size
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    logging.warning(f"Skipping {filepath}: File too large (>{MAX_FILE_SIZE/1024:.0f}KB)")
                    continue
                python_files.append(filepath)
    return python_files

def get_changed_python_files():
    """Gets a list of changed Python files using git diff."""
    result = subprocess.run(["git", "diff", "--name-only", "--diff-filter=M", "--cached", "--", "*.py"], capture_output=True, text=True) 
    if result.returncode != 0:
        result = subprocess.run(["git", "diff", "--name-only", "--diff-filter=M", "--", "*.py"], capture_output=True, text=True) 
        if result.returncode != 0:
            print(f"Error getting changed files: {result.stderr}")
            return []

    files = result.stdout.strip().split('\n')
    return [f for f in files if f]

def get_file_content(filepath):
    """Reads the content of a file."""
    with open(filepath, "r") as f:
        return f.read()

def validate_python_code(code: str) -> bool:
    """Validate if the code is syntactically correct Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def create_backup(filepath: str) -> None:
    """Create a backup of the file before modification."""
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"{Path(filepath).name}.{int(time.time())}.bak"
    Path(filepath).copy(backup_path)

def calculate_similarity(original: str, modified: str) -> float:
    """Calculate similarity ratio between original and modified code."""
    import difflib
    return difflib.SequenceMatcher(None, original, modified).ratio()

@retry_on_error()
def send_to_ollama(code: str, prompt_prefix: str = """Review this Python code for readability, PEP 8 compliance, and basic optimizations.
Return the complete improved Python code as a single block.
Focus on clarity, conciseness, and best practices.
If no improvements are needed, return the original code.

```python
"""): 
    """Sends code to Ollama using the ollama library and gets AI suggestions."""
    try:
        response: ChatResponse = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_prefix + code + "\n```",
                },
            ]
        )
        return response.message.content.strip()

    except Exception as e: 
        print(f"Error communicating with Ollama: {e}")
        return None

def process_single_file(filepath: str, stats: ProcessingStats) -> None:
    """Process a single file with all safety checks."""
    try:
        original_code = get_file_content(filepath)
        if not original_code or not validate_python_code(original_code):
            logging.warning(f"Invalid Python code in {filepath}")
            stats.files_skipped += 1
            return

        ai_suggestions = send_to_ollama(original_code)
        if not ai_suggestions or not validate_python_code(ai_suggestions):
            logging.warning(f"Invalid AI suggestions for {filepath}")
            stats.files_skipped += 1
            return

        similarity = calculate_similarity(original_code, ai_suggestions)
        if similarity > SIMILARITY_THRESHOLD:
            logging.info(f"Changes too minor for {filepath} (similarity: {similarity:.2f})")
            stats.files_skipped += 1
            return

        create_backup(filepath)
        apply_suggestions(filepath, original_code, ai_suggestions)
        stats.files_modified += 1

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        stats.errors += 1

def apply_suggestions(filepath: str, original_code: str, suggestions: str) -> None:
    """Applies AI suggestions with improved error handling and validation."""
    if not suggestions or not validate_python_code(suggestions):
        logging.warning(f"Invalid suggestions for {filepath}")
        return

    try:
        dmp = diff_match_patch()
        diffs = dmp.diff_main(original_code, suggestions)
        dmp.diff_cleanupSemantic(diffs)
        patches = dmp.patch_make(diffs)
        
        # First verify patch in memory
        patched_code, results = dmp.patch_apply(patches, original_code)
        if not all(results) or not validate_python_code(patched_code):
            logging.error(f"Invalid patch generated for {filepath}")
            return

        # Write patch file for review
        patch_text = dmp.patch_toText(patches)
        with open(PATCH_OUTPUT_FILE, "a", encoding='utf-8') as f_patch_out:
            f_patch_out.write(f"\n=== {filepath} ===\n{patch_text}\n")

        # Apply changes
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(patched_code)

        logging.info(f"Successfully applied AI suggestions to {filepath}")

    except Exception as e:
        logging.error(f"Error applying suggestions to {filepath}: {e}")
        raise

if __name__ == "__main__":
    stats = ProcessingStats()
    files_queue = deque(get_all_python_files())
    total_files = len(files_queue)
    logging.info(f"Found {total_files} Python files to process")

    if os.path.exists(PATCH_OUTPUT_FILE):
        os.remove(PATCH_OUTPUT_FILE)

    BACKUP_DIR.mkdir(exist_ok=True)
    
    if total_files > 1 and MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while files_queue:
                filepath = files_queue.popleft()
                executor.submit(process_single_file, filepath, stats)
    else:
        while files_queue:
            filepath = files_queue.popleft()
            process_single_file(filepath, stats)

    stats.print_summary()
    
    if stats.files_modified > 0:
        logging.info(f"Backups stored in: {BACKUP_DIR}")
    logging.info("AI Autofix and Optimization process completed.")