 # parser_service/core/image_embedder.py
import torch
from PIL import Image
import logging
from ..models import ModelManager

logger = logging.getLogger(__name__)

class ImageEmbedder:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.clip_model, self.clip_processor = self.model_manager.get_clip_model_and_processor()
        self.device = self.model_manager.device

    def get_image_embedding(self, image_path: str):
        """
        Generates an embedding for the given image using CLIP.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: The image embedding, or None if generation fails.
        """
        if not self.clip_model or not self.clip_processor:
            logger.error("CLIP model or processor not available for image embedding.")
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode(): # Ensure no gradients are calculated
                inputs = self.clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) # Normalize
            logger.info(f"Generated embedding for image: {image_path}")
            return image_features.cpu() # Move to CPU for general use
        except FileNotFoundError:
            logger.error(f"Image file not found for embedding: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error generating embedding for image {image_path}: {e}", exc_info=True)
            return None

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This example assumes models are downloaded and paths in config are correct.
    # It also needs a dummy image file to run.
    dummy_image_path = "dummy_image_for_embedding.png"
    import os

    try:
        Image.new('RGB', (60, 30), color = 'blue').save(dummy_image_path)
        
        mm = ModelManager() # Initialize model manager
        image_embedder = ImageEmbedder(model_manager=mm)
        
        embedding = image_embedder.get_image_embedding(dummy_image_path)
        
        if embedding is not None:
            logger.info(f"Embedding for {dummy_image_path} (shape: {embedding.shape}):\n{embedding}")
        else:
            logger.error(f"Failed to generate embedding for {dummy_image_path}.")
            
    except Exception as e:
        logger.error(f"Failed to run ImageEmbedder example: {e}", exc_info=True)
        logger.error("Ensure your ModelManager can load models and Pillow is installed.")
    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path) 