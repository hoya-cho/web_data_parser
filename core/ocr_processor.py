import torch
import logging
import paddle # For paddle.device.cuda.empty_cache()
from ..models import ModelManager
import os # For example usage
from PIL import Image, ImageDraw # For example usage
import difflib

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.ocr_model = self.model_manager.get_ocr_model()
        self.exaone_model, self.exaone_tokenizer = self.model_manager.get_exaone_model_and_tokenizer()
        self.device = self.model_manager.device # Device for EXAONE model

    def extract_ocr_text_from_image(self, image_path: str):
        """
        Extracts text from an image using PaddleOCR, similar to original script's subprocess logic.
        Returns empty string if total OCR text length is less than 10 characters.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: A single string of joined OCR results, or an empty string if no text/error or total length < 10.
        """
        if not self.ocr_model:
            logger.error("OCR model (PaddleOCR) not available.")
            return ""

        try:
            # Original script calls ocr_engine.ocr(image_path, cls=False)
            # and then paddle.device.cuda.empty_cache()
            results = self.ocr_model.ocr(image_path, cls=False) # cls=False as per original
            if self.device.startswith("cuda") and torch.cuda.is_available(): # Check to avoid error if not on CUDA
                 paddle.device.cuda.empty_cache() # As per original script

            if not results or results[0] is None:
                logger.info(f"No text found by OCR in image: {image_path}")
                return ""

            # Original script: text = ' '.join([line[1][0] for line in results[0]])
            text_lines = []
            for line_info in results[0]:
                if line_info and len(line_info) >= 2:
                    text_tuple = line_info[1]
                    if text_tuple and isinstance(text_tuple, tuple) and len(text_tuple) > 0:
                        text = text_tuple[0]
                        text_lines.append(text)
            
            joined_text = ' '.join(text_lines)
            
            # Check total length of OCR results
            if len(joined_text.strip()) < 10:
                logger.info(f"Total OCR text length ({len(joined_text.strip())}) is less than 10 characters for {image_path}")
                return ""
                
            logger.info(f"Extracted OCR text (joined) from {image_path}. Length: {len(joined_text)}")
            return joined_text
        
        except FileNotFoundError:
            logger.error(f"Image file not found for OCR: {image_path}")
            return ""
        except Exception as e:
            logger.error(f"Error during OCR extraction for image {image_path}: {e}", exc_info=True)
            return ""

    def is_similar(self, a, b, threshold=0.3):
        # threshold: 0~1, 1에 가까울수록 더 비슷해야 True
        return difflib.SequenceMatcher(None, a, b).ratio() > threshold

    def refine_ocr_text_with_exaone(self, ocr_text: str, max_new_tokens: int = 512):
        """
        Refines the extracted OCR text using the EXAONE model, with original Korean prompt.

        Args:
            ocr_text (str): The OCR text string to refine.
            max_new_tokens (int): Max new tokens for EXAONE to generate.

        Returns:
            str: The refined text, or the original text if refinement fails or EXAONE is unavailable.
        """
        if not ocr_text.strip() or len(ocr_text.strip()) < 4:
            return ocr_text
            
        if not self.exaone_model or not self.exaone_tokenizer:
            logger.warning("EXAONE model/tokenizer not available for OCR refinement. Returning raw OCR text.")
            return ocr_text
        
        # Korean prompt from original product_data_pipeline.py
        prompt = f"""아래는 한국어 OCR로 추출된 텍스트입니다.
- 잘못 인식된 글자, 오탈자, 띄어쓰기 오류를 자연스럽게 수정해주세요.
- 원본 의미를 바꾸지 마세요.

[OCR 텍스트]
{ocr_text}

[수정된 텍스트]"""

        try:
            # Note: original script used add_special_tokens=False for EXAONE inputs.
            # AutoProcessor for EXAONE might handle this differently than AutoTokenizer.
            # We'll stick to the tokenizer's default unless issues arise.
            inputs = self.exaone_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.exaone_model.device)
            
            # Original script generation parameters for EXAONE
            outputs = self.exaone_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.95,
                do_sample=True # Original script had do_sample=True
            )
            
            # Original script: result = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[수정된 텍스트]")[-1].strip()
            # The prompt itself is part of the output, so we need to remove it carefully.
            # Decoding the full output then splitting is safer.
            full_decoded_text = self.exaone_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            refined_text_marker = "[수정된 텍스트]"
            if refined_text_marker in full_decoded_text:
                refined_text = full_decoded_text.split(refined_text_marker, 1)[-1].strip()
            else:
                # If marker is not found, it might mean the model output something unexpected
                # or the prompt was not fully included. We take text after the original prompt.
                prompt_end_index = full_decoded_text.rfind(ocr_text) + len(ocr_text) if ocr_text in full_decoded_text else -1
                if prompt_end_index != -1 and prompt_end_index < len(full_decoded_text):
                    refined_text = full_decoded_text[prompt_end_index:].strip()
                else: # Fallback, could be risky
                    refined_text = full_decoded_text # Or consider it failed
                    logger.warning(f"Could not reliably find '[수정된 텍스트]' marker in EXAONE output. Output: {full_decoded_text}")

            # Cleanup if the refined text is just a repetition of the prompt structure
            if refined_text.startswith("아래는 한국어 OCR로 추출된 텍스트입니다") or refined_text == ocr_text :
                 logger.warning("EXAONE refinement might have failed or repeated input. Returning original OCR text.")
                 return ocr_text

            # refined_text가 ocr_text와 너무 다르면 원본 반환
            if not self.is_similar(refined_text, ocr_text, threshold=0.2):
                logger.warning("Refined text is not similar to OCR text. Returning original OCR text.")
                return ocr_text

            logger.info(f"Refined OCR text with EXAONE. Original length: {len(ocr_text)}, Refined length: {len(refined_text)}")
            return refined_text if refined_text.strip() else ocr_text # Return original if refinement is empty
        
        except Exception as e:
            logger.error(f"Error during EXAONE OCR text refinement: {e}", exc_info=True)
            return ocr_text # Fallback to original text

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running OCRProcessor example with updated prompts...")
    dummy_image_path = "dummy_ocr_image_for_refinement.png"

    try:
        # Create a dummy image with some sample Korean text (Pillow might need a Kfont)
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        try:
            # Attempt to use a common Korean font if available, or default
            # font = ImageFont.truetype("NanumGothic.ttf", 20) # Example
            draw.text((10, 10), "안녕 하세요. OCR 테스트 입니다.", fill='black')
            draw.text((10, 40), "이것은 오타와 함깨있는 문장입미다.", fill='black')
        except Exception as font_e:
            logger.warning(f"Could not draw text with specific font for dummy OCR image: {font_e}. Using default.")
            draw.text((10,10), "Default Font OCR Test", fill='black')
        img.save(dummy_image_path)
        logger.info(f"Created dummy OCR image: {dummy_image_path}")

        mm = ModelManager()
        ocr_processor = OCRProcessor(model_manager=mm)

        raw_ocr = ocr_processor.extract_ocr_text_from_image(dummy_image_path)
        logger.info(f"\nRaw Extracted OCR Text:\n{raw_ocr}")

        if raw_ocr:
            refined_text = ocr_processor.refine_ocr_text_with_exaone(raw_ocr)
            logger.info(f"\nRefined OCR Text (EXAONE):\n{refined_text}")
        else:
            logger.info("No raw OCR text extracted to refine.")

    except Exception as e:
        logger.error(f"Failed to run OCRProcessor example: {e}", exc_info=True)
    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            logger.info(f"Cleaned up dummy OCR image: {dummy_image_path}") 