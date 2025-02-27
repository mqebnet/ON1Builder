import os
import subprocess
from diff_match_patch import diff_match_patch
from ollama import chat, ChatResponse 
from collections import deque

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b")
PATCH_OUTPUT_FILE = "ai_patch_output.txt" 

def get_all_python_files():
    """Recursively finds all Python files in the repository, excluding the .git directory."""
    python_files = []
    for root, dirs, files in os.walk(os.getcwd()):
        if '.git' in dirs:
            dirs.remove('.git')
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def get_file_content(filepath):
    """Reads the content of a file safely with UTF-8 encoding."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def send_to_ollama(code, prompt_prefix="""Review this Python code for readability, PEP 8 compliance, and basic optimizations.
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


def apply_suggestions(filepath, original_code, suggestions):
    """Applies AI suggestions using diff-match-patch and writes patch to file."""
    if not suggestions:
        print(f"No suggestions received from AI for {filepath}, skipping patching.")
        return

    try:
        dmp = diff_match_patch()
        diffs = dmp.diff_main(original_code, suggestions)
        dmp.diff_cleanupSemantic(diffs)
        patches = dmp.patch_make(diffs)
        patch_text = dmp.patch_toText(patches)

        with open(PATCH_OUTPUT_FILE, "w", encoding="utf-8") as f_patch_out:
            f_patch_out.write(patch_text)

        print(f"\n--- Generated Patch for {filepath} (Review in 'Display Patch Output' step of workflow!) ---")

        patched_code, results = dmp.patch_apply(patches, original_code)
        if not all(results):
            print(f"Warning: Not all patches applied successfully for {filepath}.")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(patched_code)

        print(f"Successfully applied AI suggestions to {filepath} using patch.")

    except Exception as e:
        print(f"Error applying AI suggestions to {filepath}: {e}")
        print(f"Please review AI suggestions manually for {filepath}.")



if __name__ == "__main__":
    files_queue = deque(get_all_python_files())
    print(f"Found {len(files_queue)} Python files.")

    if os.path.exists(PATCH_OUTPUT_FILE):
        os.remove(PATCH_OUTPUT_FILE)

    while files_queue:
        filepath = files_queue.popleft()
        print(f"\nProcessing file: {filepath}")
        original_code = get_file_content(filepath)
        if original_code:
            ai_suggestions = send_to_ollama(original_code)
            if ai_suggestions:
                apply_suggestions(filepath, original_code, ai_suggestions)
            else:
                print(f"No suggestions received from AI for {filepath}")
        else:
            print(f"Skipping {filepath} due to file read error.")

    print("\nAI Autofix and Optimization process completed.")