import os
import json
import logging
import tempfile
import shutil
import uuid # For UID generation like in original script
import time # For duration
from datetime import datetime # For crawled_at
from urllib.parse import urlparse
import gc # For garbage collection as in original
import torch # For torch.cuda.empty_cache()
from PIL import Image
import cv2
import numpy as np

from .config import DEFAULT_RESULTS_DIR
from .models import ModelManager
from .core import (
    WebScraper,
    TextProcessor,
    ImageClassifier,
    ImageEmbedder,
    OCRProcessor,
    TableDetector,
    TableParser
)

logger = logging.getLogger(__name__)

# Utility: filter out small images and split long images
MIN_WIDTH = 100
MIN_HEIGHT = 100
MAX_SPLIT_HEIGHT = 2000  # threshold for splitting long images
MAX_SEGMENT_HEIGHT = 1500  # maximum height for split segments

def find_good_breaks(image, threshold=10, min_gap=80):
    """
    연속된 동일 색상의 라인을 찾아 자연스러운 분할 경계를 반환합니다.
    """
    image_np = np.array(image)
    height, width, _ = image_np.shape
    good_breaks = []
    prev_complex = True
    for y in range(1, height):
        line = image_np[y]
        diff = np.abs(line - line[0])  # 기준 픽셀과의 차이
        is_monochrome = np.all(diff < threshold)
        if is_monochrome and prev_complex:
            if not good_breaks or (y - good_breaks[-1]) > min_gap:
                good_breaks.append(y)
        prev_complex = not is_monochrome
    return good_breaks

def filter_breaks(breaks, image_height, max_segment_height=MAX_SEGMENT_HEIGHT):
    """
    각 구간이 max_segment_height 이하가 되도록 break를 선택합니다.
    """
    if not breaks:
        return [image_height]
    filtered = []
    last_y = 0
    for b in breaks:
        if (b - last_y) > max_segment_height:
            candidate = [br for br in breaks if last_y < br <= last_y + max_segment_height]
            if candidate:
                best_cut = candidate[-1]
                filtered.append(best_cut)
                last_y = best_cut
        elif (b - last_y) >= int(0.7 * max_segment_height):  # 충분히 큰 경우 자름
            filtered.append(b)
            last_y = b
    if image_height - last_y > 0:
        filtered.append(image_height)
    return filtered

def split_image(image, breaks, save_dir, prefix):
    """
    실제로 이미지를 분할하여 파일로 저장합니다.
    """
    output_paths = []
    prev_y = 0
    for idx, y in enumerate(breaks):
        cropped = image.crop((0, prev_y, image.width, y))
        output_path = os.path.join(save_dir, f"{prefix}_{idx+1}.png")
        cropped.save(output_path)
        output_paths.append(output_path)
        prev_y = y
    return output_paths

def filter_and_split_images(image_paths, min_width=MIN_WIDTH, min_height=MIN_HEIGHT, max_height=MAX_SPLIT_HEIGHT):
    filtered_paths = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if width < min_width or height < min_height:
                    continue  # Skip small images
                if height > max_height:
                    # Find natural break points
                    breaks = find_good_breaks(img)
                    filtered_breaks = filter_breaks(breaks, height, MAX_SEGMENT_HEIGHT)
                    # Split image
                    base, ext = os.path.splitext(img_path)
                    split_paths = split_image(img, filtered_breaks, os.path.dirname(img_path), os.path.basename(base))
                    # Filter out small splits
                    for split_path in split_paths:
                        with Image.open(split_path) as split_img:
                            w, h = split_img.size
                            if w >= min_width and h >= min_height:
                                filtered_paths.append(split_path)
                else:
                    filtered_paths.append(img_path)
        except Exception as e:
            continue
    return filtered_paths

class ProductParsingPipeline:
    def __init__(
        self,
        model_manager: ModelManager,
        results_base_dir: str = DEFAULT_RESULTS_DIR,
        embed_product_photos: bool = True,
        save_intermediate_results: bool = False,
        save_final_results: bool = True  # Add new parameter
    ):
        self.model_manager = model_manager
        self.results_base_dir = results_base_dir
        self.embed_product_photos = embed_product_photos
        self.save_intermediate_results = save_intermediate_results
        self.save_final_results = save_final_results  # Store the new parameter
        self.pipeline_uid = str(uuid.uuid4())[:8]
        os.makedirs(self.results_base_dir, exist_ok=True)

        # Core components initialization
        self.text_processor = TextProcessor(model_manager=self.model_manager)
        self.image_classifier = ImageClassifier(model_manager=self.model_manager)
        self.image_embedder = ImageEmbedder(model_manager=self.model_manager)
        self.ocr_processor = OCRProcessor(model_manager=self.model_manager)
        logger.info("ProductParsingPipeline initialized with shared processors.")

    def _save_intermediate_file(self, source_path: str, target_dir: str, filename: str = None) -> str:
        """중간 결과물을 저장하는 헬퍼 함수"""
        if not self.save_intermediate_results:
            return source_path
            
        os.makedirs(target_dir, exist_ok=True)
        if filename is None:
            filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(source_path, target_path)
        return target_path

    def process_url(self, url: str, progress_callback=None):
        start_process_time = time.time()
        logger.info(f"[{self.pipeline_uid}] Starting processing for URL: {url}")
        
        if progress_callback:
            progress_callback("Initialization", 0.0, {"status": "Starting pipeline"})
        
        # Create output directories if saving intermediate results
        if self.save_intermediate_results:
            url_specific_output_dir = os.path.join(self.results_base_dir, f"url_{self.pipeline_uid}")
            os.makedirs(url_specific_output_dir, exist_ok=True)
            images_dir = os.path.join(url_specific_output_dir, "images")
            ocr_dir = os.path.join(url_specific_output_dir, "ocr")
            tables_dir = os.path.join(url_specific_output_dir, "tables")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(ocr_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            logger.info(f"[{self.pipeline_uid}] Results will be saved in: {url_specific_output_dir}")
        # else:
        #     url_specific_output_dir = None
        if self.save_final_results:
            url_specific_output_dir = os.path.join(self.results_base_dir, f"url_{self.pipeline_uid}")
            os.makedirs(url_specific_output_dir, exist_ok=True)
            logger.info(f"[{self.pipeline_uid}] Results will be saved in: {url_specific_output_dir}")

        # Temporary directories for intermediate files
        with tempfile.TemporaryDirectory(prefix=f"parser_{self.pipeline_uid}_", suffix="_downloads") as temp_download_dir, \
             tempfile.TemporaryDirectory(prefix=f"parser_{self.pipeline_uid}_", suffix="_table_crops") as temp_table_crop_dir:
            
            logger.info(f"[{self.pipeline_uid}] Using temp download dir: {temp_download_dir}")
            logger.info(f"[{self.pipeline_uid}] Using temp table crop dir: {temp_table_crop_dir}")

            # Initialize components
            web_scraper = WebScraper(download_dir=temp_download_dir) 
            table_detector = TableDetector(model_manager=self.model_manager, crop_output_dir=temp_table_crop_dir)
            table_parser = TableParser(model_manager=self.model_manager)

            # --- 1. Web Scraping ---
            if progress_callback:
                progress_callback("Web Scraping", 0.1, {"status": "Starting web scraping"})
            
            scraped_data = web_scraper.fetch_text_and_images(url)
            if not scraped_data or not scraped_data[0]:
                logger.error(f"[{self.pipeline_uid}] Failed to scrape URL: {url}. Aborting.")
                if progress_callback:
                    progress_callback("Web Scraping", 1.0, {"status": "Failed", "error": "Failed to scrape URL"})
                return None
            
            text_content_file, downloaded_image_paths, saved_image_urls = scraped_data
            if progress_callback:
                progress_callback("Web Scraping", 1.0, {
                    "status": "Completed",
                    "text_file": text_content_file,
                    "image_count": len(downloaded_image_paths)
                })

            # Save scraped text if needed
            if self.save_intermediate_results:
                scraped_text_path = os.path.join(url_specific_output_dir, "scraped_text.txt")
                shutil.copy2(text_content_file, scraped_text_path)
                text_content_file = scraped_text_path

            # --- Filter and split images ---
            if progress_callback:
                progress_callback("Image Processing", 0.2, {"status": "Filtering and splitting images"})
            
            filtered_image_paths = []
            split_image_url_map = {}
            for orig_path, orig_url in zip(downloaded_image_paths, saved_image_urls):
                split_paths = filter_and_split_images([orig_path])
                for sp in split_paths:
                    if self.save_intermediate_results:
                        new_path = self._save_intermediate_file(sp, images_dir)
                        filtered_image_paths.append(new_path)
                    else:
                        filtered_image_paths.append(sp)
                    split_image_url_map[filtered_image_paths[-1]] = orig_url

            if progress_callback:
                progress_callback("Image Processing", 0.3, {
                    "status": "Completed",
                    "filtered_image_count": len(filtered_image_paths)
                })

            # Read raw text content
            raw_text_content = ""
            try:
                with open(text_content_file, "r", encoding="utf-8") as f:
                    raw_text_content = f.read()
            except Exception as e:
                logger.error(f"[{self.pipeline_uid}] Failed to read scraped text file {text_content_file}: {e}")

            # --- 2. Text Processing ---
            if progress_callback:
                progress_callback("Text Processing", 0.4, {"status": "Extracting product information"})
            
            summary_text = self.text_processor.extract_product_info_from_text(raw_text_content)
            if summary_text is None:
                logger.warning(f"[{self.pipeline_uid}] Product info extraction failed for {url}. Using empty string.")
                summary_text = ""

            if progress_callback:
                progress_callback("Text Processing", 0.5, {
                    "status": "Completed",
                    "summary_length": len(summary_text)
                })

            # Initialize result containers
            product_clip_embedding_info = {"img_url": None, "clip_embedding": None}
            images_ocr_texts = []
            tables_natural_language_texts = []
            image_path_to_categories = {}

            # --- 3. Image Classification ---
            if progress_callback:
                progress_callback("Image Classification", 0.6, {"status": "Classifying images"})
            
            for idx, img_path in enumerate(filtered_image_paths):
                if not os.path.exists(img_path):
                    logger.warning(f"[{self.pipeline_uid}] Image path {img_path} does not exist. Skipping.")
                    continue
                raw_category_list = self.image_classifier.classify_image_categories(img_path)
                image_path_to_categories[img_path] = raw_category_list
                
                if progress_callback:
                    progress = 0.6 + (0.1 * (idx + 1) / len(filtered_image_paths))
                    progress_callback("Image Classification", progress, {
                        "status": "In progress",
                        "current_image": idx + 1,
                        "total_images": len(filtered_image_paths)
                    })

            if progress_callback:
                progress_callback("Image Classification", 0.7, {
                    "status": "Completed",
                    "classified_images": image_path_to_categories
                })

            # Save image classifications if needed
            if self.save_intermediate_results:
                classifications_file = os.path.join(url_specific_output_dir, "image_classifications.json")
                with open(classifications_file, 'w') as f:
                    json.dump(image_path_to_categories, f, indent=2)

            # --- 3a. CLIP Embedding ---
            if self.embed_product_photos:
                if progress_callback:
                    progress_callback("CLIP Embedding", 0.7, {"status": "Generating CLIP embeddings"})
                
                images_embedding = []
                for img_path, categories in image_path_to_categories.items():
                    if "product photo" in categories:
                        clip_embedding_tensor = self.image_embedder.get_image_embedding(img_path)
                        if clip_embedding_tensor is not None:
                            img_url = split_image_url_map.get(img_path, None)
                            images_embedding.append({
                                "img_url": img_url,
                                "clip_embedding": clip_embedding_tensor.tolist()
                            })
                
                if progress_callback:
                    progress_callback("CLIP Embedding", 0.8, {
                        "status": "Completed",
                        "embedded_images": len(images_embedding)
                    })
            else:
                images_embedding = None

            # --- 3b. Table Detection & Parsing --- 
            if progress_callback:
                progress_callback("Table Processing", 0.8, {"status": "Detecting and parsing tables"})
            
            # Track which images were processed as tables
            processed_as_tables = set()
            
            for img_path, categories in image_path_to_categories.items():
                if "table with text" in categories:
                    try:
                        cropped_table_paths = table_detector.detect_and_crop_tables(img_path)
                        
                        if not cropped_table_paths:
                            # Process as text-only image with OCR only if no tables detected
                            raw_ocr_text = self.ocr_processor.extract_ocr_text_from_image(img_path)
                            if raw_ocr_text and raw_ocr_text.strip():
                                refined_text = self.ocr_processor.refine_ocr_text_with_exaone(raw_ocr_text)
                                if refined_text and refined_text.strip():
                                    images_ocr_texts.append(refined_text)
                                else:
                                    images_ocr_texts.append(raw_ocr_text)
                            continue

                        # If tables are detected, process them with table parser
                        processed_as_tables.add(img_path)  # Mark this image as processed by table parser
                        for crop_path in cropped_table_paths:
                            if not os.path.exists(crop_path):
                                continue
                            
                            # Save table crops if needed
                            if self.save_intermediate_results:
                                crop_filename = os.path.basename(crop_path)
                                crop_path = self._save_intermediate_file(crop_path, tables_dir, crop_filename)
                            
                            nl_table_text = table_parser.parse_table_image_to_natural_language(crop_path)
                            if nl_table_text:
                                tables_natural_language_texts.append(nl_table_text)
                                
                                # Save parsed table text if needed
                                if self.save_intermediate_results:
                                    parsed_text_path = os.path.join(tables_dir, f"{os.path.splitext(crop_filename)[0]}_parsed.txt")
                                    with open(parsed_text_path, 'w') as f:
                                        f.write(nl_table_text)
                            
                    except Exception as e:
                        logger.error(f"[{self.pipeline_uid}] Error processing table/text image {img_path}: {e}")
                        continue

            if progress_callback:
                progress_callback("Table Processing", 0.9, {
                    "status": "Completed",
                    "parsed_tables": len(tables_natural_language_texts)
                })

            # --- 3c. OCR and Refinement --- 
            # Skip OCR processing for images that were successfully processed as tables
            if progress_callback:
                progress_callback("OCR Processing", 0.9, {"status": "Performing OCR and refinement"})
            
            for img_path, categories in image_path_to_categories.items():
                # Skip images that were already processed as tables
                if img_path in processed_as_tables:
                    continue
                    
                # Process only text-only images and product with text images
                if "text-only image" in categories or "product with text" in categories:
                    raw_ocr_text = self.ocr_processor.extract_ocr_text_from_image(img_path)
                    if raw_ocr_text and raw_ocr_text.strip():
                        refined_text = self.ocr_processor.refine_ocr_text_with_exaone(raw_ocr_text)
                        
                        # Save OCR results if needed
                        if self.save_intermediate_results:
                            base_name = os.path.splitext(os.path.basename(img_path))[0]
                            raw_ocr_path = os.path.join(ocr_dir, f"{base_name}_raw.txt")
                            refined_ocr_path = os.path.join(ocr_dir, f"{base_name}_refined.txt")
                            
                            with open(raw_ocr_path, 'w') as f:
                                f.write(raw_ocr_text)
                            if refined_text and refined_text.strip():
                                with open(refined_ocr_path, 'w') as f:
                                    f.write(refined_text)
                        
                        if refined_text and refined_text.strip():
                            images_ocr_texts.append(refined_text)
                        else:
                            images_ocr_texts.append(raw_ocr_text)

            if progress_callback:
                progress_callback("OCR Processing", 0.95, {
                    "status": "Completed",
                    "processed_images": len(images_ocr_texts)
                })

            # --- 4. Structure and Save Results --- 
            if progress_callback:
                progress_callback("Finalizing", 0.95, {"status": "Structuring final results"})
            
            final_result_structure = {
                "meta": {
                    "source_url": url,
                    "crawled_at": datetime.now().isoformat(),
                    "duration_sec": round(time.time() - start_process_time, 2),
                    "uuid": self.pipeline_uid
                },
                "product_text_info": {
                    "summary": summary_text
                },
                "product_images_info": {
                    "images_embedding": images_embedding,
                    "images_text": images_ocr_texts,
                    "tables_text": tables_natural_language_texts
                }
            }
            
            # Save final JSON
            if self.save_final_results:  # Use the new parameter
                output_json_path = os.path.join(url_specific_output_dir, "structured_product_data.json")
                try:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(final_result_structure, f, ensure_ascii=False, indent=2)
                    logger.info(f"[{self.pipeline_uid}] Successfully saved structured data to {output_json_path}")
                except Exception as e:
                    logger.error(f"[{self.pipeline_uid}] Failed to save JSON results: {e}", exc_info=True)
            
            if progress_callback:
                progress_callback("Finalizing", 1.0, {
                    "status": "Completed",
                    "final_result": final_result_structure
                })
            
            return final_result_structure

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting ProductParsingPipeline (updated prompts) example...")
    try:
        model_manager = ModelManager()
        # save_intermediate_results=True로 설정하여 중간 결과물 저장
        pipeline = ProductParsingPipeline(model_manager=model_manager, save_intermediate_results=True)
        test_url = "https://www.ikea.com/kr/ko/p/malm-ottoman-bed-white-20404807/"

        logger.info(f"Running pipeline for URL: {test_url}")
        result_data = pipeline.process_url(test_url)
        if result_data:
            logger.info(f"Pipeline completed successfully for {test_url}. UUID: {result_data['meta']['uuid']}")
        else:
            logger.error(f"Pipeline failed for {test_url}.")

    except Exception as e:
        logger.error(f"Error during pipeline example execution: {e}", exc_info=True)
    finally:
        logger.info("ProductParsingPipeline (updated prompts) example finished.") 