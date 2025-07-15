import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig
)
from paddleocr import PaddleOCR
from ultralytics import YOLO
import logging
import os
from ..config import (
    GEMMA_MODEL_ID, EXAONE_MODEL_ID, CLIP_MODEL_ID, 
    YOLO_MODEL_PATH, DEVICE, OFFLOAD_FOLDER, OCR_LANG
)
import gc

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")

        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

        self.gemma_processor = None
        self.gemma_model = None
        self.exaone_tokenizer = None
        self.exaone_model = None
        self.clip_processor = None
        self.clip_model = None
        self.yolo_model = None
        self.ocr_model = None

        self._load_models()

    def _get_device(self):
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            return DEVICE
        logger.warning(f"CUDA device {DEVICE} not available or torch not built with CUDA. Falling back to CPU.")
        return "cpu"

    def _load_models(self):
        logger.info("Loading models...")
        try:
            # Gemma Model (using AutoProcessor and Gemma3ForConditionalGeneration)
            self.gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            self.gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                GEMMA_MODEL_ID,
                device_map="auto",
                offload_folder=OFFLOAD_FOLDER
            ).eval()
            logger.info(f"Gemma model ({GEMMA_MODEL_ID}) loaded with AutoProcessor.")

            # EXAONE Model (with BitsAndBytesConfig as per original script)
            bnb_config_exaone = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.exaone_tokenizer = AutoProcessor.from_pretrained(EXAONE_MODEL_ID, trust_remote_code=True)
            self.exaone_model = AutoModelForCausalLM.from_pretrained(
                EXAONE_MODEL_ID,
                quantization_config=bnb_config_exaone,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            logger.info(f"EXAONE model ({EXAONE_MODEL_ID}) loaded.")

            # CLIP Model
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
            logger.info(f"CLIP model ({CLIP_MODEL_ID}) loaded.")

            # YOLO Model
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}.")
            dummy_img_path = self._create_dummy_image()
            if dummy_img_path:
                try:
                    self.yolo_model(dummy_img_path, device=self.device, verbose=False)
                    logger.info("YOLO model initialized on device.")
                except Exception as e:
                    logger.error(f"Error during YOLO dummy prediction: {e}")
                finally:
                    if os.path.exists(dummy_img_path):
                        os.remove(dummy_img_path)
            
            # PaddleOCR Model
            self.ocr_model = self.get_ocr_model()

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise

    def _create_dummy_image(self):
        try:
            from PIL import Image
            import numpy as np
            dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(dummy_array)
            dummy_path = "./dummy_yolo_init.png"
            img.save(dummy_path)
            return dummy_path
        except ImportError:
            logger.warning("Pillow not installed. Cannot create dummy image for YOLO initialization.")
            return None
        except Exception as e:
            logger.error(f"Failed to create dummy image: {e}")
            return None

    def get_gemma_model_and_processor(self):
        if not self.gemma_model or not self.gemma_processor:
            logger.error("Gemma model/processor not loaded. This should not happen if initialization was successful.")
            try:
                self.gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
                self.gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                    GEMMA_MODEL_ID, device_map="auto", offload_folder=OFFLOAD_FOLDER).eval()
                logger.info("Gemma model reloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to reload Gemma model: {e}")
                return None, None
        return self.gemma_model, self.gemma_processor

    def get_exaone_model_and_tokenizer(self):
        if not self.exaone_model or not self.exaone_tokenizer:
            logger.error("EXAONE model/tokenizer not loaded.")
            try:
                bnb_config_exaone = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                self.exaone_tokenizer = AutoProcessor.from_pretrained(EXAONE_MODEL_ID, trust_remote_code=True)
                self.exaone_model = AutoModelForCausalLM.from_pretrained(
                    EXAONE_MODEL_ID, quantization_config=bnb_config_exaone,
                    torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
                logger.info("EXAONE model reloaded successfully.")
            except Exception as e:
                 logger.error(f"Failed to reload EXAONE model: {e}")
                 return None, None
        return self.exaone_model, self.exaone_tokenizer

    def get_clip_model_and_processor(self):
        if not self.clip_model or not self.clip_processor:
            logger.error("CLIP model/processor not loaded.")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
                logger.info("CLIP model reloaded successfully.")
            except Exception as e:
                 logger.error(f"Failed to reload CLIP model: {e}")
                 return None, None
        return self.clip_model, self.clip_processor

    def get_yolo_model(self):
        if not self.yolo_model:
            logger.error("YOLO model not loaded.")
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logger.info("YOLO model reloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to reload YOLO model: {e}")
                return None
        return self.yolo_model

    def get_ocr_model(self):
        if self.ocr_model is None:
            logger.info("OCR model is not loaded. Loading PaddleOCR...")
            try:
                # Check CUDA availability more strictly
                use_gpu_flag = False
                if self.device.startswith("cuda"):
                    try:
                        import paddle
                        if paddle.device.is_compiled_with_cuda():
                            use_gpu_flag = True
                            logger.info("PaddlePaddle CUDA support confirmed.")
                        else:
                            logger.warning("PaddlePaddle was not compiled with CUDA support. Falling back to CPU.")
                    except ImportError:
                        logger.warning("PaddlePaddle not available. Falling back to CPU.")
                    except Exception as e:
                        logger.warning(f"Error checking CUDA support: {e}. Falling back to CPU.")

                # Initialize PaddleOCR with explicit CPU fallback
                self.ocr_model = PaddleOCR(
                    lang=OCR_LANG,
                    use_angle_cls=True,
                    use_gpu=use_gpu_flag,
                    show_log=False,
                    enable_mkldnn=True  # Enable MKL-DNN for better CPU performance
                )
                logger.info(f"PaddleOCR model loaded for language: {OCR_LANG}. use_gpu={use_gpu_flag}")
            except Exception as e:
                logger.error(f"Failed to load OCR model: {e}", exc_info=True)
                # Ensure ocr_model remains None if loading fails
                self.ocr_model = None 
                return None
        return self.ocr_model

    def unload_ocr_model(self):
        if self.ocr_model is not None:
            logger.info("Unloading OCR model to free up memory...")
            try:
                # PaddleOCR object itself doesn't have a specific 'unload' or 'delete' method in its public API.
                # Deleting the reference and relying on Python's garbage collector is the standard way.
                del self.ocr_model
                self.ocr_model = None
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache (both Paddle and PyTorch, just in case)
                try:
                    import paddle
                    if hasattr(paddle, 'device') and hasattr(paddle.device, 'cuda') and hasattr(paddle.device.cuda, 'empty_cache'):
                        if paddle.device.is_compiled_with_cuda():
                            paddle.device.cuda.empty_cache()
                            logger.info("Paddle CUDA cache cleared after OCR unload.")
                except ImportError:
                    pass
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("PyTorch CUDA cache cleared after OCR unload.")
                logger.info("OCR model unloaded and memory cleanup attempted.")
            except Exception as e:
                logger.error(f"Error during OCR model unload: {e}", exc_info=True)
        else:
            logger.info("OCR model was not loaded, no need to unload.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Attempting to initialize ModelManager...")
    try:
        model_manager = ModelManager()
        logger.info(f"ModelManager initialized. Device: {model_manager.device}")

        gemma_model, gemma_processor = model_manager.get_gemma_model_and_processor()
        if gemma_model and gemma_processor:
            logger.info("Successfully retrieved Gemma model and processor.")
        else:
            logger.error("Failed to retrieve Gemma model/processor.")

        model_manager2 = ModelManager()
        if model_manager is model_manager2:
            logger.info("ModelManager is a singleton.")
        else:
            logger.error("ModelManager is NOT a singleton.")

    except Exception as e:
        logger.error(f"Error in ModelManager example: {e}", exc_info=True) 