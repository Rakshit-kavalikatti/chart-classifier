from packages import *  # Import all required modules from your packages

# ==========================================================
# 🔧 Setup script: create config.json with model_path
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this setup script
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")    # Path to store config.json

# Determine model save directory dynamically
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "chart_classifier")

# Create configuration dictionary
config_data = {
    "model_path": MODEL_SAVE_DIR
}

# ==========================================================
# ⚡ Create config.json if it doesn't exist
# This ensures that the model path is stored for future use
# ==========================================================
if not os.path.exists(CONFIG_PATH):
    # Write the config data to file
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"✅ config.json created at {CONFIG_PATH}")
    print(f"Model path set to: {MODEL_SAVE_DIR}")
else:
    print(f"⚠️ config.json already exists at {CONFIG_PATH}")
    print(f"Model path: {MODEL_SAVE_DIR}")

# ==========================================================
# ⬇️ Function: download_model()
# Downloads and saves the model + processor locally
# Only needed once if model doesn't already exist
# ==========================================================
def download_model():
    """Download model + processor once (requires internet)."""
    print("⬇️ Downloading model... Please wait...")

    # Create the folder for model files if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Download processor and model from Hugging Face
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

    # Save the processor and model locally for offline use
    processor.save_pretrained(MODEL_SAVE_DIR)
    model.save_pretrained(MODEL_SAVE_DIR)
    print("✅ Model downloaded successfully and saved locally.\n")

# ==========================================================
# ⬇️ Function: create_chart_command()
# Sets up a global terminal command 'chart' to run predict_chart.py
# Works for both Windows and Linux/macOS
# ==========================================================
def create_chart_command():
    repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the repository
    target_script = os.path.join(repo_dir, "predict_chart.py")  # Script to run

    # Find Python executable path
    python_exe = sys.executable

    # Ensure the model is downloaded first
    download_model()

    if platform.system() == "Windows":
        # Windows: create a .bat file in a folder that is in PATH
        script_dir = os.path.expanduser(r"~\AppData\Local\Microsoft\WindowsApps")
        command_path = os.path.join(script_dir, "chart.bat")

        # Create the .bat file that runs the Python script with any arguments
        with open(command_path, "w") as f:
            f.write(f'@echo off\n"{python_exe}" "{target_script}" %*\n')

        print(f"✅ Command 'chart' is now available globally in your terminal.\n")
        print(f"➡️  Try: chart C:\\path\\to\\image.png")

    else:
        # Linux/macOS: create a shell script in ~/.local/bin
        script_dir = os.path.expanduser("~/.local/bin")
        os.makedirs(script_dir, exist_ok=True)
        command_path = os.path.join(script_dir, "chart")

        # Write shell script
        with open(command_path, "w") as f:
            f.write(f'#!/bin/bash\n"{python_exe}" "{target_script}" "$@"\n')
        os.chmod(command_path, 0o755)  # Make executable

        print(f"✅ Command 'chart' installed successfully.")
        print(f"➡️  Try: chart /path/to/image.png")

# ==========================================================
# 🚀 Entry point: run the setup to create global command
# ==========================================================
if __name__ == "__main__":
    create_chart_command()
