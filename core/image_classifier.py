import torch
from PIL import Image, UnidentifiedImageError
import logging
import ast # For ast.literal_eval as in original script
import os
from ..models import ModelManager

logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gemma_model, self.gemma_processor = self.model_manager.get_gemma_model_and_processor()
        # self.device = self.model_manager.device # model.device used directly from gemma_model

    def classify_image_categories(self, image_path: str, max_new_tokens: int = 64):
        """
        Classifies an image into predefined categories using Gemma, based on the original script's logic.
        The model is expected to return a Python list of category strings.

        Args:
            image_path (str): Path to the image file.
            max_new_tokens (int): Max new tokens for Gemma to generate for the category list.

        Returns:
            list: A list of category strings (e.g., ["product photo", "text-only image"]).
                  Returns ["unidentified image"] if image opening fails.
                  Returns ["classification_failed"] or a raw string in a list if parsing fails.
        """
        if not self.gemma_model or not self.gemma_processor:
            logger.error("Gemma model or processor not available for image classification.")
            return ["classification_setup_failed"]

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prompt from original product_data_pipeline.py
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an assistant that classifies images based on their content."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image}, # This is how AutoProcessor expects image data
                    {"type": "text", "text": (
                    "Classify this image into one of the following categories:\n\n"
                    "1. product photo — an image showing the product itself, such as clothes, accessories, shoes, etc.\n"
                    "2. table with text — an image containing a structured table, like a size chart, ingredient list, or comparison table.\n"
                    "3. text-only image — an image containing only descriptive text, such as usage instructions, delivery info, or notices.\n"
                    "4. unrelated graphic — any image that does not contain meaningful product information, such as banners, icons, logos, or advertisements.\n"
                    "5. product with text — an image showing the product itself together with text.(text is visible in the image).\n\n"
                    "Please respond with only the category name.\n"
                    "Rules:\n"
                    "- The result must be a **list of one or more categories**.\n"
                    "- If 'unrelated graphic' is selected, no other category can be selected.\n"
                    "- 'table with text' and 'text-only image' **cannot appear together**.\n"
                    "- 'product photo' can appear **alone** or **together** with either 'table with text' or 'text-only image'.\n"
                    "- 'product with text' can appear alone or together with 'table with text', but not with 'unrelated graphic' or 'text-only image' or 'product photo'.\n\n"
                    "Respond only with a Python list of category names, e.g.:\n"
                    "[\"product with text\"]\n"
                    "[\"product photo\", \"text-only image\"]"
                )}
                ]}
            ]

            # Ensure inputs are on the same device as the model expects (and correct dtype if needed)
            # GemmaForConditionalGeneration with device_map="auto" should handle internal device placement.
            # Inputs should be moved to the model's primary device or a device it can access.
            inputs = self.gemma_processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.gemma_model.device, dtype=torch.bfloat16) # Original script used bfloat16 for inputs here

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation_output = self.gemma_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # As per original
                    eos_token_id=self.gemma_processor.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            decoded_categories_str = self.gemma_processor.decode(generation_output[0][input_len:], skip_special_tokens=True)
            logger.debug(f"Raw classification output for {image_path}: {decoded_categories_str}")

            # Parse the string output into a list (as per original script)
            try:
                # Ensure the string is a valid Python literal for a list
                # Basic cleaning: sometimes models add extra quotes or text around the list
                clean_str = decoded_categories_str.strip()
                if not (clean_str.startswith("[") and clean_str.endswith("]")):
                     # Attempt to find a list-like structure if the model didn't adhere perfectly
                    if '[' in clean_str and ']' in clean_str:
                        start_index = clean_str.find('[')
                        end_index = clean_str.rfind(']') + 1
                        clean_str = clean_str[start_index:end_index]
                    else: # If no brackets, it's likely not a list, treat as single category if non-empty
                        if clean_str:
                            logger.warning(f"Classification output for {image_path} is not a list: '{clean_str}'. Treating as single category.")
                            return [clean_str] 
                        else:
                            logger.warning(f"Classification output for {image_path} is empty.")
                            return ["classification_empty_output"]

                category_list = ast.literal_eval(clean_str)
                if not isinstance(category_list, list):
                    # If ast.literal_eval results in non-list (e.g. a string if input was '"cat"')
                    logger.warning(f"Parsed category for {image_path} is not a list: {category_list}. Wrapping in list.")
                    category_list = [str(category_list)] # Convert to string and wrap

                # If 'table with text' is detected, remove 'product with text' from categories
                if "table with text" in category_list and "product with text" in category_list:
                    logger.info(f"Removing 'product with text' category as 'table with text' is detected for {image_path}")
                    category_list.remove("product with text")

            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse category list string from LLM for {image_path}: '{decoded_categories_str}'. Error: {e}. Returning raw string as a category.")
                # Fallback: return the raw string as a single-element list if parsing fails
                category_list = [decoded_categories_str.strip()] if decoded_categories_str.strip() else ["classification_parse_failed"]
            
            logger.info(f"Image {image_path} classified as: {category_list}")
            return category_list

        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to open or identify image {image_path}: {e}")
            return ["unidentified_image"] # As per original script
        except Exception as e:
            logger.error(f"Error during image classification for {image_path}: {e}", exc_info=True)
            return ["classification_error"]

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running ImageClassifier example with updated prompts...")
    
    dummy_image_path = "dummy_classification_image.png"
    try:
        # Create a dummy image
        Image.new('RGB', (100, 100), color='blue').save(dummy_image_path)
        logger.info(f"Created dummy image: {dummy_image_path}")

        mm = ModelManager()
        image_classifier = ImageClassifier(model_manager=mm)
        
        categories = image_classifier.classify_image_categories(dummy_image_path)
        logger.info(f"Image {dummy_image_path} classified categories: {categories}")
        # Expected: Gemma might not produce a perfect list for a dummy blue image without fine-tuning.
        # The goal here is to test the flow and parsing logic.

    except Exception as e:
        logger.error(f"Failed to run ImageClassifier example: {e}", exc_info=True)
    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            logger.info(f"Cleaned up dummy image: {dummy_image_path}") 