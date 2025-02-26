import os
import subprocess
import json
from diff_match_patch import diff_match_patch
from ollama import chat, ChatResponse 

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b")
PATCH_OUTPUT_FILE = "ai_patch_output.txt" 

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
        patch_text = dmp.diff_toPatch(diffs)

        with open(PATCH_OUTPUT_FILE, "w") as f_patch_out:
            f_patch_out.write(patch_text)

        print(f"\n--- Generated Patch for {filepath} (Review in 'Display Patch Output' step of workflow!) ---")

        patches = dmp.patch_fromText(patch_text)
        patched_code, _ = dmp.patch_apply(patches, original_code)

        with open(filepath, "w") as f:
            f.write(patched_code)

        print(f"Successfully applied AI suggestions to {filepath} using patch.")

    except Exception as e:
        print(f"Error applying AI suggestions to {filepath}: {e}")
        print(f"Please review AI suggestions manually for {filepath}.")



if __name__ == "__main__":
    changed_files = get_changed_python_files()
    print(f"Changed Python files: {changed_files}")

    if os.path.exists(PATCH_OUTPUT_FILE):
        os.remove(PATCH_OUTPUT_FILE)

    for filepath in changed_files:
        print(f"\nProcessing file: {filepath}")
        original_code = get_file_content(filepath)
        ai_suggestions = send_to_ollama(original_code)

        if ai_suggestions:
            apply_suggestions(filepath, original_code, ai_suggestions)
        else:
            print(f"No suggestions received from AI for {filepath}")

    print("\nAI Autofix and Optimization process completed.")