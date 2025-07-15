# parser_service_updated_prompts/config.py

# Model IDs
GEMMA_MODEL_ID = "google/gemma-3-4b-it" # Used for text summary, image classification, table parsing
EXAONE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" # Used for OCR refinement
CLIP_MODEL_ID = "openai/clip-vit-large-patch14" # Used for image embedding
YOLO_MODEL_PATH = "/app/parser_service_updated_prompts/yolo_pth/yolov10x_best.pt" # Path to YOLO model (ensure this path is correct)

# Device configuration
DEVICE = "cuda:0" # Default to "cuda:0". Will fall back to "cpu" if CUDA is not available.
OFFLOAD_FOLDER = "./offload_models" # For HuggingFace model offloading

# OCR settings
OCR_LANG = 'korean'

# Default directory for final structured results (one subdir per URL)
DEFAULT_RESULTS_DIR = "./parsing_results_updated"

# Concurrency settings for main.py runner
MAX_CONCURRENT_WORKERS = 2 # Adjust based on system resources (CPU, GPU VRAM) 