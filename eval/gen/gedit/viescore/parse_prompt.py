import os

def create_python_file_with_texts(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("# This file is generated automatically through parse_prompt.py\n\n")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    var_name = "_" + file_path.replace(folder_path, "").replace(os.sep, "_").replace(".txt", "").strip("_")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().replace('"""', '\"\"\"')
                        out_file.write(f'{var_name} = """{content}"""\n\n')

# Example usage
current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)
folder_path = os.path.join(current_folder_path, "prompts_raw")
output_file = os.path.join(current_folder_path, "vie_prompts.py")
create_python_file_with_texts(folder_path, output_file)
