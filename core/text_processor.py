import torch
import logging
from ..models import ModelManager

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gemma_model, self.gemma_processor = self.model_manager.get_gemma_model_and_processor()
        self.device = self.model_manager.device

    def extract_product_info_from_text(self, text_content: str, max_text_len: int = 1024, max_new_tokens: int = 1024):
        """
        Extracts product information from raw webpage text using Gemma, 
        based on the original script's logic.

        Args:
            text_content (str): The raw text scraped from the product page.
            max_text_len (int): Max length for truncating input text to Gemma.
            max_new_tokens (int): Max new tokens for Gemma to generate.

        Returns:
            str: Extracted product information string, or None if failed.
        """
        if not self.gemma_model or not self.gemma_processor:
            logger.error("Gemma model or processor not available for text processing.")
            return None
        if not text_content or not text_content.strip():
            logger.warning("No text content provided for product info extraction.")
            return ""

        try:
            # Truncate text as per original script
            tokenized = self.gemma_processor.tokenizer(
                text_content,
                return_tensors="pt",
                truncation=True,
                max_length=max_text_len,
                add_special_tokens=False
            )
            truncated_text = self.gemma_processor.tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a smart assistant that extracts product information from raw webpage text."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        "The following text was scraped from an e-commerce product page. "
                        "Please extract only the information that is directly related to the product being sold.\n"
                        "Focus on fields such as product name, price, shipping, discount, options, review count, rating, and availability period.\n\n"
                        f"{truncated_text}"
                    )}]
                }
            ]

            # Ensure model is on the correct device for inputs. 
            # `device_map="auto"` in ModelManager should handle model parts, 
            # but inputs still need to be moved if device is specified and different from inputs' current device.
            # The processor itself might not place tensors on device if gemma_model.device is not directly accessible like that.
            # A common pattern is to move inputs to model.device if model is a single nn.Module
            # For device_map="auto", model.device might be a bit complex. Processor should handle it.

            # Original script uses torch.bfloat16 for inputs. 
            # This should be compatible if the model is loaded with bfloat16 or can cast.
            inputs = self.gemma_processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.gemma_model.device, dtype=torch.bfloat16) # Ensure inputs are on the same device and dtype as model expects
            
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = self.gemma_model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False, # As per original
                    eos_token_id=self.gemma_processor.tokenizer.eos_token_id
                )
            
            decoded_info = self.gemma_processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            logger.info(f"Extracted product info. Output length: {len(decoded_info)}")
            # print(f"\n✅ Extracted Product Summary Info:\n{decoded_info}") # For debugging, as in original
            return decoded_info

        except Exception as e:
            logger.error(f"Error during product info extraction: {e}", exc_info=True)
            return None

# Example (similar to original TextProcessor, but using the new method name and logic)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running TextProcessor example with updated prompts...")
    try:
        mm = ModelManager() 
        text_processor = TextProcessor(model_manager=mm)
        
        sample_text = (
            "[상품명] 고효율 저소음 공기청정기 AQ-1000\n" 
            "[가격] 299,000원 (할인 적용 시 250,000원)\n" 
            "[배송] 무료배송 (제주/도서산간 5,000원 추가)\n" 
            "[옵션] 1. 화이트 2. 블랙 (+10,000원)\n" 
            "[리뷰] 1,234개 | 평점 4.8/5.0\n" 
            "[구매혜택] 포토 리뷰 작성 시 스타벅스 기프티콘 증정 (선착순 100명)\n" 
            "이 제품은 최신 헤파 필터 기술을 사용하여 초미세먼지까지 99.9% 제거합니다. " 
            "수면 모드에서는 20dB 이하의 저소음으로 작동하여 편안한 잠자리를 제공합니다. " 
            "지금 구매하시고 깨끗한 공기를 경험하세요!"
        )
        
        extracted_info = text_processor.extract_product_info_from_text(sample_text)
        
        if extracted_info:
            logger.info("Original Text (sample):")
            logger.info(sample_text)
            logger.info("\nExtracted Product Info:")
            logger.info(extracted_info)
        else:
            logger.error("Failed to extract product info.")
            
    except Exception as e:
        logger.error(f"Failed to run TextProcessor example: {e}", exc_info=True) 