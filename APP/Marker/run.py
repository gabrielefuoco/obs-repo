import os
import shutil

def move_markdown_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip root directory
        
        for file in files:
            if file.endswith(".md"):
                src_path = os.path.join(subdir, file)
                dest_path = os.path.join(root_dir, file)
                
                # Ensure the file is not overwritten
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(root_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    move_markdown_files(script_dir)
