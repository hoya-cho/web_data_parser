import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import pandas as pd
from datetime import datetime
import glob # For finding files
import sys
from pathlib import Path

# Get the project root directory (two levels up from this file)
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 직접 경로로 임포트
from parser_service_updated_prompts.pipeline import ProductParsingPipeline
from parser_service_updated_prompts.models import ModelManager
from parser_service_updated_prompts.config import DEFAULT_RESULTS_DIR

# Initialize ModelManager once
# Using st.cache_resource for newer Streamlit versions to ensure it's loaded once
@st.cache_resource
def get_model_manager():
    st.info("Initializing ModelManager... (This may take a moment on first run)")
    manager = ModelManager()
    st.success("ModelManager initialized.")
    return manager

def get_pipeline(_model_manager):
    """Create a new pipeline instance for each parsing"""
    return ProductParsingPipeline(
        model_manager=_model_manager,
        results_base_dir=DEFAULT_RESULTS_DIR,
        save_intermediate_results=True,  # 웹 데모에서는 중간 결과물 저장 활성화
        save_final_results=True  # 웹 데모에서는 최종 결과물 저장 활성화
    )

def display_image_with_caption(image_path, caption, width=300):
    try:
        # 경로 정규화
        image_path = os.path.normpath(image_path)
        image = Image.open(image_path)
        st.image(image, caption=caption, width=width)
    except FileNotFoundError:
        st.warning(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Could not load image {image_path}: {e}")

def display_text_content(file_path, header):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        st.subheader(header)
        st.text_area("Content", content, height=200)
    except FileNotFoundError:
        st.warning(f"Text file not found: {file_path}")
    except Exception as e:
        st.error(f"Could not read text file {file_path}: {e}")

def visualize_pipeline_steps(output_dir_path, final_json_result):
    st.header("Pipeline Execution Details")

    # 0. Meta Information (already in final_json_result)
    with st.expander("0. Meta Information", expanded=True):
        meta = final_json_result.get("meta", {})
        st.write(f"**Source URL**: {meta.get('source_url', 'N/A')}")
        st.write(f"**Pipeline UUID**: {meta.get('uuid', 'N/A')}")
        st.write(f"**Crawled At**: {meta.get('crawled_at', 'N/A')}")
        st.write(f"**Duration (seconds)**: {meta.get('duration_sec', 'N/A')}")

    # 1. Text + Image Crawling
    with st.expander("1. Text + Image Crawling Results", expanded=False):
        st.subheader("Scraped Text")
        scraped_text_file = os.path.join(output_dir_path, "scraped_text.txt")
        if os.path.exists(scraped_text_file):
            display_text_content(scraped_text_file, "View Scraped Text")
        else:
            st.info("Scraped text file not found (expected at scraped_text.txt in output directory). Displaying summary from final JSON if available.")
            summary = final_json_result.get("product_text_info", {}).get("summary")
            if summary:
                st.text_area("Product Summary (from final JSON as fallback)", summary, height=150)

        st.subheader("Downloaded Images")
        images_dir = os.path.join(output_dir_path, "images")
        if os.path.isdir(images_dir):
            downloaded_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if downloaded_images:
                # Display images in columns for better layout
                cols = st.columns(3) 
                for i, img_path in enumerate(downloaded_images):
                    with cols[i % 3]:
                        display_image_with_caption(img_path, os.path.basename(img_path), width=200)
            else:
                st.info("No images found in the 'images' subfolder.")
        else:
            st.info("Images subfolder not found.")

    # 2. Text -> Product Info Summary
    with st.expander("2. Text -> Product Info Summary", expanded=False):
        summary_text = final_json_result.get("product_text_info", {}).get("summary", "Summary not available.")
        st.text_area("LLM Summarized Product Information", summary_text, height=150)

    # 3. Image Classification (Gemma)
    with st.expander("3. Image Classification Results", expanded=False):
        classifications_file = os.path.join(output_dir_path, "image_classifications.json")
        images_base_path = os.path.join(output_dir_path, "images")
        if os.path.exists(classifications_file) and os.path.isdir(images_base_path):
            with open(classifications_file, 'r') as f:
                image_classifications = json.load(f)
            
            if image_classifications:
                for img_filename, categories in image_classifications.items():
                    st.markdown(f"**Image:** `{img_filename}`")
                    # 이미지 경로 정규화
                    img_full_path = os.path.normpath(os.path.join(images_base_path, os.path.basename(img_filename)))
                    col1, col2 = st.columns([1,2])
                    with col1:
                        display_image_with_caption(img_full_path, os.path.basename(img_filename), width=150)
                    with col2:
                        st.write("**Categories:**")
                        st.json(categories)
                    st.divider()
            else:
                st.info("No image classification data found.")
        else:
            st.info("Image classifications file or images folder not found.")

    # 4. CLIP Embedding (Optional)
    with st.expander("4. CLIP Embedding Results", expanded=False):
        embeddings_info = final_json_result.get("product_images_info", {}).get("images_embedding")
        if embeddings_info:
            st.write(f"Found {len(embeddings_info)} image(s) with CLIP embeddings.")
            for idx, embed_item in enumerate(embeddings_info):
                st.markdown(f"**Embedded Image {idx+1}**")
                if embed_item.get("img_url"):
                     # Try to find local path if URL is a filename, otherwise show URL
                    img_url = embed_item["img_url"]
                    st.image(img_url, caption=f"Source: {img_url}", width=200)
                
                embedding_vector = embed_item.get("clip_embedding", [])
                if embedding_vector:
                    st.text_area(f"CLIP Embedding (first 10 values of {len(embedding_vector)})", 
                                 str(embedding_vector[:10]) + "...", height=100)
                st.divider()
        else:
            st.info("No CLIP embedding information in final JSON result.")

    # 5. Text Image OCR Refinement
    with st.expander("5. Text Image OCR Refinement", expanded=False):
        ocr_dir = os.path.join(output_dir_path, "ocr")
        images_base_path = os.path.join(output_dir_path, "images") # Assuming original images are here
        
        # We need a way to link OCR files to original images.
        # Let's assume OCR files are named like 'image_filename_raw.txt' and 'image_filename_refined.txt'
        if os.path.isdir(ocr_dir) and os.path.isdir(images_base_path):
            ocr_files_raw = glob.glob(os.path.join(ocr_dir, "*_raw.txt"))
            if not ocr_files_raw:
                st.info("No raw OCR text files found in 'ocr' subfolder.")

            for raw_file_path in ocr_files_raw:
                base_name = os.path.basename(raw_file_path).replace("_raw.txt", "")
                refined_file_path = os.path.join(ocr_dir, f"{base_name}_refined.txt")
                # Try to find the corresponding image. This logic might need refinement based on actual filenames.
                # For simplicity, assume base_name is the image filename (e.g., "image1.png" -> "image1_raw.txt")
                
                # Find a matching image file (could be .png, .jpg, etc.)
                original_image_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_image_path = os.path.join(images_base_path, base_name + ext)
                    if os.path.exists(potential_image_path):
                        original_image_path = potential_image_path
                        break
                
                st.markdown(f"**OCR for image:** `{base_name}`")
                if original_image_path:
                    display_image_with_caption(original_image_path, f"Original for {base_name}", width=200)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Raw OCR Text")
                    try:
                        with open(raw_file_path, 'r') as f: st.text_area("Raw", f.read(), height=150, key=f"raw_{base_name}")
                    except FileNotFoundError: st.warning("Raw OCR file not found.")
                
                with col2:
                    st.subheader("Refined OCR Text (EXAONE)")
                    try:
                        with open(refined_file_path, 'r') as f: st.text_area("Refined", f.read(), height=150, key=f"refined_{base_name}")
                    except FileNotFoundError: st.info(f"Refined OCR file not found for {base_name}. Raw OCR may have been used or refinement failed.")
                st.divider()
        else:
            st.info("'ocr' or 'images' subfolder not found for OCR results.")

    # 6. Table Image Detection + Crop & 7. Cropped Table LLM Analysis
    with st.expander("6. Table Image Detection + Crop & Cropped Table LLM Analysis", expanded=False):
        tables_dir = os.path.join(output_dir_path, "tables")
        if os.path.isdir(tables_dir):
            # Find all table crop images with the correct pattern
            cropped_table_images = sorted(glob.glob(os.path.join(tables_dir, "*_table_crop_*.jpg")))
            
            if not cropped_table_images:
                st.info("No table images found.")

            # Group cropped images by their original image
            original_images = {}
            for crop_path in cropped_table_images:
                # Extract original image name from crop filename (e.g., "image_025_1" from "image_025_1_table_crop_00.jpg")
                original_name = os.path.basename(crop_path).split("_table_crop_")[0]
                if original_name not in original_images:
                    original_images[original_name] = []
                original_images[original_name].append(crop_path)

            # Display tables for each original image
            for original_name, crop_paths in original_images.items():
                st.markdown(f"**Tables detected in image:** `{original_name}`")
                
                st.subheader("Cropped Tables & Parsed Text")
                for crop_img_path in crop_paths:
                    crop_basename = os.path.basename(crop_img_path).replace(".jpg", "")
                    parsed_text_file = os.path.join(tables_dir, f"{crop_basename}_parsed.txt")
                    
                    col1, col2 = st.columns([1,2])
                    with col1:
                        display_image_with_caption(crop_img_path, f"Cropped: {crop_basename}", width=200)
                    with col2:
                        if os.path.exists(parsed_text_file):
                            with open(parsed_text_file, 'r') as f:
                                st.text_area("LLM Parsed Text from Table", f.read(), height=100, key=f"parsed_{crop_basename}")
                        else:
                            st.info("Parsed text not found for this crop.")
                    st.markdown("---") # Small divider for each crop
                st.divider() # Divider for each original image with tables
        else:
            st.info("'tables' subfolder not found for table processing results.")
            # Fallback to final JSON if available
            tables_text_final = final_json_result.get("product_images_info", {}).get("tables_text", [])
            if tables_text_final:
                st.subheader("Parsed Table Texts (from final JSON)")
                for i, text in enumerate(tables_text_final):
                    st.text_area(f"Table Text {i+1}", text, height=100)


    # 8. Final JSON Result
    with st.expander("8. Final JSON Result", expanded=True):
        st.json(final_json_result)
        
        # Download button for the final JSON
        pipeline_uuid = final_json_result.get("meta", {}).get("uuid", "unknown_uuid")
        json_string = json.dumps(final_json_result, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Full JSON Result",
            data=json_string,
            file_name=f"parsed_product_data_{pipeline_uuid}.json",
            mime="application/json",
        )

def main():
    st.set_page_config(layout="wide", page_title="Product Parser Demo")
    st.title("🛍️ Product Parser Pipeline Demo")
    st.markdown("Enter a product page URL to parse its content and visualize the pipeline steps.")

    model_manager = get_model_manager()
    pipeline = get_pipeline(model_manager)

    # Session state to store results and progress
    if "pipeline_ran" not in st.session_state:
        st.session_state.pipeline_ran = False
        st.session_state.final_json_result = None
        st.session_state.output_dir_path = None
        st.session_state.error_message = None
        st.session_state.current_step = None
        st.session_state.step_progress = {}
        st.session_state.intermediate_results = {}
        st.session_state.step_containers = {}

    with st.sidebar:
        st.header("Input URL")
        url_to_parse = st.text_input("Product URL:", placeholder="https://example.com/product-page")
        
        if st.button("🚀 Parse Product URL", type="primary"):
            if url_to_parse:
                # Reset session state
                st.session_state.pipeline_ran = False
                st.session_state.final_json_result = None
                st.session_state.output_dir_path = None
                st.session_state.error_message = None
                st.session_state.current_step = None
                st.session_state.step_progress = {}
                st.session_state.intermediate_results = {}
                st.session_state.step_containers = {}

                # Create main progress indicator
                progress_placeholder = st.empty()
                step_placeholder = st.empty()

                with st.spinner(f"🤖 Parsing URL: {url_to_parse}... This may take a few minutes."):
                    try:
                        start_time = datetime.now()
                        
                        # Define progress callback
                        def update_progress(step_name, progress, results=None):
                            # Update progress bar
                            progress_placeholder.progress(progress)
                            step_placeholder.text(f"Current step: {step_name}")
                            
                            # Store results in session state
                            st.session_state.current_step = step_name
                            st.session_state.step_progress[step_name] = progress
                            if results:
                                st.session_state.intermediate_results[step_name] = results
                            
                            # Create or update step container
                            if step_name not in st.session_state.step_containers:
                                st.session_state.step_containers[step_name] = st.empty()
                            
                            # Update step container with current results
                            with st.session_state.step_containers[step_name].container():
                                st.subheader(f"Step: {step_name}")
                                
                                if results:
                                    if step_name == "Web Scraping":
                                        if "text_file" in results:
                                            st.write("Scraped Text File:", results["text_file"])
                                        if "image_count" in results:
                                            st.write(f"Downloaded Images: {results['image_count']}")
                                    
                                    elif step_name == "Image Processing":
                                        if "filtered_image_count" in results:
                                            st.write(f"Filtered Images: {results['filtered_image_count']}")
                                    
                                    elif step_name == "Text Processing":
                                        if "summary_length" in results:
                                            st.write(f"Summary Length: {results['summary_length']} characters")
                                    
                                    elif step_name == "Image Classification":
                                        if "classified_images" in results:
                                            st.write(f"Classified {len(results['classified_images'])} images")
                                    
                                    elif step_name == "CLIP Embedding":
                                        if "embedded_images" in results:
                                            st.write(f"Embedded Images: {results['embedded_images']}")
                                    
                                    elif step_name == "Table Processing":
                                        if "parsed_tables" in results:
                                            st.write(f"Parsed Tables: {results['parsed_tables']}")
                                    
                                    elif step_name == "OCR Processing":
                                        if "processed_images" in results:
                                            st.write(f"Processed Images: {results['processed_images']}")
                                    
                                    elif step_name == "Finalizing":
                                        if "final_result" in results:
                                            st.write("Final result structure created successfully")

                        # Run pipeline with progress callback
                        final_json = pipeline.process_url(url_to_parse, progress_callback=update_progress)
                        
                        if final_json and "meta" in final_json and "uuid" in final_json["meta"]:
                            pipeline_uid = final_json["meta"]["uuid"]
                            output_dir = os.path.join(pipeline.results_base_dir, f"url_{pipeline_uid}")

                            st.session_state.final_json_result = final_json
                            st.session_state.output_dir_path = output_dir
                            st.session_state.pipeline_ran = True
                            st.session_state.error_message = None
                            
                            end_time = datetime.now()
                            st.success(f"✅ Parsing successful in {(end_time - start_time).total_seconds():.2f} seconds!")
                            st.balloons()
                        else:
                            st.session_state.error_message = "Parsing completed, but the result format is unexpected or missing metadata."
                            st.error(st.session_state.error_message)

                    except Exception as e:
                        st.session_state.error_message = f"An error occurred during parsing: {str(e)}"
                        st.error(st.session_state.error_message)
                        st.exception(e)
            else:
                st.warning("Please enter a URL to parse.")

    # Show final results if pipeline has completed
    if st.session_state.pipeline_ran and st.session_state.final_json_result and st.session_state.output_dir_path:
        if os.path.isdir(st.session_state.output_dir_path):
            visualize_pipeline_steps(st.session_state.output_dir_path, st.session_state.final_json_result)
        else:
            st.error(f"Output directory not found: {st.session_state.output_dir_path}. Cannot display detailed steps. Check pipeline saving logic.")
            st.subheader("Final JSON Result (if available)")
            st.json(st.session_state.final_json_result)
    elif st.session_state.pipeline_ran and st.session_state.final_json_result and not st.session_state.output_dir_path:
        st.warning("Pipeline ran, final JSON is available, but output directory path for artifacts is missing.")
        st.subheader("Final JSON Result")
        st.json(st.session_state.final_json_result)


if __name__ == "__main__":
    main() 