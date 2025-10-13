from packages import *  # Import all necessary modules from your custom packages

# ==========================================================
# ⚙️ Load configuration
# The path to the model will be loaded from a JSON config file
# This allows the model path to be dynamic for different users
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")    # Config file path

# Load the JSON config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Get only the value of 'model_path' to use as SAVE_DIR
SAVE_DIR = config["model_path"]


# ==========================================================
# 🧠 Function: classify_chart()
# Loads the model and classifies the given chart image
# ==========================================================
def classify_chart(image_path, model_path):
    """Run chart classification offline."""

    # Load the model and processor from local storage
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode (no training required)

    # Load the input image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")  # Prepare inputs for the model

    # Disable gradient computation for faster and memory-efficient inference
    with torch.no_grad():
        outputs = model(**inputs)

        # Compute class probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the class with the highest probability
        confidence, idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        label = model.config.id2label[idx.item()]  # Get the class label

    # Clean the label text (remove "_chart" suffix)
    normalized = label.lower().replace("_chart", "")

    # Prepare the JSON result (can be extended with emojis if desired)
    result = {
        "predicted_class": normalized,
        "confidence_score": round(confidence, 3)
    }

    # Print the JSON output neatly
    print(json.dumps(result, ensure_ascii=False, indent=4))

# ==========================================================
# 🚀 Entry point: Script execution starts here
# ==========================================================
if __name__ == "__main__":

    # Ensure the user provides at least one argument (image path)
    if len(sys.argv) < 2:
        print("Usage: python predict_chart.py <image_path>")
        sys.exit(1)

    # Combine all command-line arguments into a single string
    # This allows users to paste paths with spaces without quotes
    img_path = " ".join(sys.argv[1:])

    # Run the chart classification
    classify_chart(img_path, SAVE_DIR)
