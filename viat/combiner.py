import os

# Configuration: The name of the file to generate
OUTPUT_FILENAME = "full_project_source.txt"

# Extensions or files to ignore (to keep output clean)
IGNORE_EXTENSIONS = {'.pyc', '.exe', '.dll', '.so', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.git', '__pycache__'}
IGNORE_FILES = {OUTPUT_FILENAME, 'combine_files.py'}

def combine_project():
    # Get the current directory where the script is running
    root_dir = os.getcwd()
    
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as outfile:
        # Walk through the directory tree
        for foldername, subfolders, filenames in os.walk(root_dir):
            
            # Remove hidden/system folders from the walk (optional)
            subfolders[:] = [d for d in subfolders if not d.startswith('.')]

            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                
                # Get the relative path (e.g., "src/main.py" instead of "C:/Users/.../src/main.py")
                relative_path = os.path.relpath(file_path, root_dir)
                
                # Check if we should skip this file
                if (filename in IGNORE_FILES or 
                    any(filename.endswith(ext) for ext in IGNORE_EXTENSIONS)):
                    continue

                try:
                    # Write the Separator
                    outfile.write("---\n")
                    
                    # Write the File Path
                    outfile.write(f"{relative_path}\n")
                    
                    # Write the File Content
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as infile:
                        content = infile.read()
                        outfile.write(content)
                    
                    # Add a newline after content for readability
                    outfile.write("\n\n")
                    
                    print(f"Processed: {relative_path}")
                    
                except Exception as e:
                    print(f"Skipping {relative_path}: {e}")

    print(f"\nSuccess! All files combined into '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    combine_project()