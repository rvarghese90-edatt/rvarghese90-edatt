import json
import os
from config.settings import OUTPUT_DIR
from src.utils import get_logger

logger = get_logger(__name__)

def save_results(results):
    output_path = os.path.join(OUTPUT_DIR, "model_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved model evaluation results to {output_path}")