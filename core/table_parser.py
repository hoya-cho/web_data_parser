import torch
from PIL import Image, UnidentifiedImageError
import logging
import os # For example
from ..models import ModelManager
# OCRProcessor is not directly used here anymore if we only parse images with Gemma VLM directly.
# However, if the strategy is to OCR the table image first, then pass text to Gemma, it would be needed.
# The original `parse_table_image` function in `product_data_pipeline.py` directly gives the image to Gemma.

logger = logging.getLogger(__name__)

class TableParser:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gemma_model, self.gemma_processor = self.model_manager.get_gemma_model_and_processor()
        # self.device = self.model_manager.device # gemma_model.device used

    def parse_table_image_to_natural_language(self, table_image_path: str, max_new_tokens: int = 512):
        """
        Parses a cropped table image directly using Gemma (VLM capabilities via AutoProcessor)
        to extract data as natural language sentences, based on the original script's logic.

        Args:
            table_image_path (str): Path to the cropped table image file.
            max_new_tokens (int): Max new tokens for Gemma to generate.

        Returns:
            str: A string of natural language sentences describing the table content, or None if parsing fails.
        """
        if not self.gemma_model or not self.gemma_processor:
            logger.error("Gemma model or processor not available for table parsing.")
            return None

        try:
            image = Image.open(table_image_path).convert("RGB")
            
            # Prompt from original product_data_pipeline.py `parse_table_image`
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a document understanding assistant."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "This image contains a table. "
                        "Extract the data as natural language sentences.\n\n"
                        "Use this format, or something similar that's concise and clear."
                        # The original prompt implies the model should infer the format.
                        # For more predictable output, a few-shot example could be added here within the text prompt.
                    )}
                ]}
            ]
            
            inputs = self.gemma_processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.gemma_model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation_output = self.gemma_model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens
                    # Original didn't specify eos_token_id here, but it's good practice
                    # eos_token_id=self.gemma_processor.tokenizer.eos_token_id 
                )
            
            parsed_text = self.gemma_processor.decode(generation_output[0][input_len:], skip_special_tokens=True).strip()
            
            logger.info(f"Parsed table image {table_image_path} into natural language. Output length: {len(parsed_text)}")
            return parsed_text

        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to open or identify table image {table_image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing table image {table_image_path}: {e}", exc_info=True)
            return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running TableParser example with updated prompts...")
    
    dummy_table_image_path = "dummy_table_for_nlp_parser.png"
    from PIL import ImageDraw, ImageFont # For creating a dummy table image

    def create_dummy_table_image_for_nlp(path):
        try:
            img = Image.new('RGB', (300, 150), color='white')
            draw = ImageDraw.Draw(img)
            try: font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except IOError: font = ImageFont.load_default()
            draw.text((10, 10), "상품 | 가격 | 재고", fill='black', font=font)
            draw.text((10, 30), "---------- | ---------- | ----", fill='black', font=font)
            draw.text((10, 50), "노트북 | 120만원 | 5개", fill='black', font=font)
            draw.text((10, 70), "키보드 | 5만원 | 20개", fill='black', font=font)
            img.save(path)
            logger.info(f"Created dummy table image: {path}")
            return True
        except Exception as e:
            logger.error(f"Error creating dummy table image: {e}")
            return False

    if create_dummy_table_image_for_nlp(dummy_table_image_path):
        try:
            mm = ModelManager()
            table_parser = TableParser(model_manager=mm)
            
            natural_language_output = table_parser.parse_table_image_to_natural_language(dummy_table_image_path)
            
            if natural_language_output:
                logger.info(f"Parsed Table (Natural Language):\n{natural_language_output}")
            else:
                logger.error("Failed to parse table to natural language.")

        except Exception as e:
            logger.error(f"Failed to run TableParser example: {e}", exc_info=True)
        finally:
            if os.path.exists(dummy_table_image_path):
                os.remove(dummy_table_image_path)
                logger.info(f"Cleaned up dummy table image: {dummy_table_image_path}")
    else:
        logger.error("Could not create dummy table image for testing TableParser.") 