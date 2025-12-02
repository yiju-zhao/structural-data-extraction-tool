
import json
import re

def update_html_with_json(html_file_path, json_file_path):
    with open(json_file_path, 'r') as f:
        new_data = json.load(f)
    
    # Use json.dumps to convert the Python list/dict to a JSON string
    # ensure_ascii=False will output actual Unicode characters instead of \uXXXX escapes,
    # which is generally safer when embedding directly into HTML/JavaScript.
    new_data_str = json.dumps(new_data, indent=2, ensure_ascii=False)

    with open(html_file_path, 'r', encoding='utf-8') as f: # Specify encoding for reading HTML
        html_content = f.read()

    # Define the pattern to find the 'allPapers = [...];' block
    # We use a non-greedy match `.*?` to ensure it doesn't match across multiple blocks
    # if they existed (though unlikely in this specific case).
    # We also capture the parts before and after the actual JSON array.
    pattern = r"(let\s+allPapers\s*=\s*)\[.*?\](;)" # Adjusted pattern to match 'let allPapers' and ending semicolon more broadly
    
    # The replacement will reconstruct the string using the captured groups
    # and insert the new_data_str
    def replacement_func(match):
        # The captured group 1 is "let allPapers = "
        # The new_data_str is the JSON array
        # The captured group 2 is ";renderPapers(allPapers);" or just ";"
        # I'm adding a specific call to renderPapers() after the assignment to ensure it always gets called.
        return f"{match.group(1)}{new_data_str}{match.group(2)}\n        renderPapers(allPapers);"

    # Perform the substitution using the callable replacement
    # re.DOTALL is important for '.' to match newlines
    updated_html_content = re.sub(pattern, replacement_func, html_content, flags=re.DOTALL)

    with open(html_file_path, 'w', encoding='utf-8') as f: # Specify encoding for writing HTML
        f.write(updated_html_content)

if __name__ == "__main__":
    html_file = "/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/visualization/index.html"
    json_file = "/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/visualization/vla/papers.json"
    update_html_with_json(html_file, json_file)
    print(f"Updated {html_file} with data from {json_file}")
