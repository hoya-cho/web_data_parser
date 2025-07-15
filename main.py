import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

from .pipeline import ProductParsingPipeline
from .models import ModelManager
from .config import MAX_CONCURRENT_WORKERS, DEFAULT_RESULTS_DIR

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Should be __name__ for logger name to be 'parser_service_updated_prompts.main'

def main():
    parser = argparse.ArgumentParser(description="Product Data Parsing Pipeline CLI (Updated Prompts)")
    parser.add_argument(
        "urls",
        nargs='+',
        help="One or more product page URLs to process."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Base directory to save parsing results. Default: {DEFAULT_RESULTS_DIR}"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_CONCURRENT_WORKERS,
        help=f"Number of concurrent workers to process URLs. Default: {MAX_CONCURRENT_WORKERS}"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more detailed output."
    )
    parser.add_argument(
        "--embed_product_photos",
        action="store_true",
        help="If set, generate and store CLIP embeddings for all product photo images. If not set, skip embedding."
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="If set, save intermediate results (scraped text, images, OCR results, etc.) to the results directory."
    )
    parser.add_argument(
        "--save_final",
        action="store_true",
        default=True,
        help="If set, save final JSON results. Default is True."
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger("parser_service_updated_prompts").setLevel(logging.DEBUG)
        for handler in logging.getLogger("parser_service_updated_prompts").handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled for 'parser_service_updated_prompts' package.")

    logger.info("Initializing ModelManager (this may take time as models are loaded)...")
    start_model_load_time = time.time()
    try:
        model_manager = ModelManager()
    except Exception as e:
        logger.error(f"Fatal error initializing ModelManager: {e}", exc_info=True)
        return
    
    end_model_load_time = time.time()
    logger.info(f"ModelManager initialized in {end_model_load_time - start_model_load_time:.2f} seconds.")
    logger.info(f"Using device: {model_manager.device}")
    if model_manager.device == "cpu":
        logger.warning("WARNING: Running on CPU. Performance will be significantly slower.")

    pipeline = ProductParsingPipeline(
        model_manager=model_manager,
        results_base_dir=args.results_dir,
        embed_product_photos=args.embed_product_photos,
        save_intermediate_results=args.save_intermediate,
        save_final_results=args.save_final
    )
    
    if args.save_intermediate:
        logger.info(f"Intermediate results will be saved in: {os.path.abspath(args.results_dir)}")
    if args.save_final:
        logger.info("Final JSON results will be saved.")
    
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    logger.info(f"Processing {len(args.urls)} URL(s) with up to {args.workers} concurrent worker(s)...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_url = {executor.submit(pipeline.process_url, url): url for url in args.urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    pipeline_uid = result.get("meta", {}).get("uuid", "unknown_uid")
                    logger.info(f"[{pipeline_uid}] Successfully completed processing for URL: {url}")
                    processed_count += 1
                    if args.save_final:
                        logger.info("Final JSON results were saved.")
                else:
                    logger.error(f"Processing failed for URL: {url} (returned None).")
                    failed_count += 1
            except Exception as exc:
                logger.error(f"URL {url} generated an unhandled exception during processing: {exc}", exc_info=True)
                failed_count += 1
    
    total_time = time.time() - start_time
    logger.info("--- Processing Summary ---")
    logger.info(f"Total URLs submitted: {len(args.urls)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed to process: {failed_count}")
    logger.info(f"Total processing time (excluding model load): {total_time:.2f} seconds")
    if processed_count > 0:
        avg_time = total_time / processed_count
        logger.info(f"Average time per successfully processed URL: {avg_time:.2f} seconds")
    logger.info(f"Results saved in base directory: {os.path.abspath(args.results_dir)}")
    if args.save_intermediate:
        logger.info("Intermediate results were saved for each processed URL.")

if __name__ == "__main__":
    # Run as: python -m parser_service_updated_prompts.main <URL1> <URL2> ...
    main() 