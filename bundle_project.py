import zipfile
import os
from pathlib import Path

def bundle_project(output_name="project_upload.zip"):
    # Define directories to include and exclude
    include_paths = [
        Path("src"),
        Path("data/processed"),
        Path("hpc")
    ]
    include_files = [
        "requirements.txt",
        "run_training.py",
        "setup.py",
        "README.md",
        "secrets.env"
    ]
    
    # Exclusion pattern (specifically for the raw data)
    exclude_prefix = os.path.join("data", "raw")

    print(f"📦 Creating {output_name}...")
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add specific directories
        for folder in include_paths:
            if not folder.exists():
                print(f"⚠️ Warning: Folder {folder} not found. Skipping.")
                continue
                
            for file in folder.rglob('*'):
                # Skip if it's in the raw data folder or a python cache
                if str(file).startswith(exclude_prefix) or "__pycache__" in str(file):
                    continue
                if file.is_file():
                    zipf.write(file, file)
                    print(f"  + {file}")

        # 2. Add individual root files
        for f_name in include_files:
            f_path = Path(f_name)
            if f_path.exists():
                zipf.write(f_path, f_path)
                print(f"  + {f_path}")

    print(f"\n✅ Done! {output_name} is ready for upload to Sockeye.")

if __name__ == "__main__":
    bundle_project()
