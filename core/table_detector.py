import logging
import os
import cv2 # For image loading and cropping
from PIL import Image # For saving cropped images
import supervision as sv # As per original script
from ..models import ModelManager

logger = logging.getLogger(__name__)

class TableDetector:
    def __init__(self, model_manager: ModelManager, crop_output_dir: str):
        self.model_manager = model_manager
        self.yolo_model = self.model_manager.get_yolo_model()
        self.device = self.model_manager.device # YOLO model can specify device on call
        self.crop_output_dir = crop_output_dir
        os.makedirs(self.crop_output_dir, exist_ok=True)
        logger.info(f"TableDetector initialized. Crop output directory: {self.crop_output_dir}")

    def detect_and_crop_tables(self, image_path: str, yolo_conf: float = 0.1, yolo_iou: float = 0.8):
        """
        Detects tables in an image using YOLO, crops them, and saves them.
        Aligns with original script's `detect_and_crop_tables`.

        Args:
            image_path (str): Path to the image file.
            yolo_conf (float): Confidence threshold for YOLO detection.
            yolo_iou (float): IoU threshold for YOLO detection.

        Returns:
            list: A list of paths to the cropped table images. 
                  Returns an empty list if no tables are detected or an error occurs.
        """
        if not self.yolo_model:
            logger.error("YOLO model not available for table detection.")
            return []

        try:
            # Original script reads with cv2 for YOLO, then uses PIL Image for cropping dimensions from YOLO results.
            # It's more straightforward to use the cv2 image for cropping if detections are from cv2 image.
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                logger.error(f"Failed to load image with OpenCV: {image_path}")
                return []

            # Perform detection using self.device (from ModelManager)
            results = self.yolo_model(image_path, conf=yolo_conf, iou=yolo_iou, device=self.device, verbose=False)
            # results is a list, get the first (and only) result object
            if not results or not results[0]:
                logger.info(f"No YOLO results for {image_path}")
                return []
            result_obj = results[0]

            # Use supervision for Detections, as in original script
            detections = sv.Detections.from_ultralytics(result_obj)
            
            # Filter for 'Table' class - original script: mask = detections.data["class_name"] == "Table"
            # We need to ensure class_name is available in detections.data or handle it.
            # sv.Detections usually has `class_id` and `confidence`. `names` map from model can be used.
            table_class_name = 'Table' # Case-sensitive, adjust if model uses 'table'
            table_class_id = None
            for cid, cname in self.yolo_model.names.items():
                if cname == table_class_name:
                    table_class_id = cid
                    break
            
            if table_class_id is None:
                logger.warning(f"Class '{table_class_name}' not found in YOLO model names: {self.yolo_model.names}. Trying lowercase 'table'.")
                table_class_name_lower = 'table'
                for cid, cname in self.yolo_model.names.items():
                    if cname == table_class_name_lower:
                        table_class_id = cid
                        table_class_name = table_class_name_lower
                        break
                if table_class_id is None:
                    logger.error(f"Class '{table_class_name}' (and lowercase) not in YOLO model names. Cannot detect tables.")
                    return []

            # Filter detections by class ID
            table_detections = detections[detections.class_id == table_class_id]
            logger.info(f"Found {len(table_detections)} potential '{table_class_name}' detections in {image_path} (after class filtering). Confidences: {table_detections.confidence}")

            cropped_table_paths = []
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            for i, (xyxy_coords) in enumerate(table_detections.xyxy):
                # Ensure xyxy_coords are integers for slicing
                x1, y1, x2, y2 = map(int, xyxy_coords)
                
                # Crop using OpenCV image (image_cv)
                # Original: crop = image[int(y1):int(y2), int(x1):int(x2)] (assuming image is cv2 image)
                cropped_table_cv = image_cv[y1:y2, x1:x2]
                
                if cropped_table_cv.size == 0:
                    logger.warning(f"Skipping empty crop for table {i} from {image_path} at coords {xyxy_coords}")
                    continue

                crop_filename = f"{base_filename}_table_crop_{i:02d}.jpg" # Original used .jpg
                crop_filepath = os.path.join(self.crop_output_dir, crop_filename)
                
                # Save the OpenCV crop
                success = cv2.imwrite(crop_filepath, cropped_table_cv)
                if success:
                    cropped_table_paths.append(crop_filepath)
                    logger.info(f"Detected and saved cropped table: {crop_filepath}")
                else:
                    logger.error(f"Failed to save cropped table: {crop_filepath}")
            
            if not cropped_table_paths:
                logger.info(f"No tables of class '{table_class_name}' saved from image: {image_path}")

            return cropped_table_paths

        except FileNotFoundError:
            logger.error(f"Image file not found for table detection: {image_path}")
            return []
        except Exception as e:
            logger.error(f"Error during table detection/cropping for {image_path}: {e}", exc_info=True)
            return []

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running TableDetector example with updated logic...")
    dummy_image_with_table_path = "dummy_table_detection_image.png"
    dummy_crop_dir = "./dummy_table_crops_updated"

    try:
        # Create a dummy image (ideally with a table-like structure for YOLO)
        img = Image.new('RGB', (600, 400), color='lightgrey')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 150], outline="black", width=2) # A box to simulate a table
        draw.text((60,60), "Fake Table", fill="black")
        img.save(dummy_image_with_table_path)
        logger.info(f"Created dummy image for table detection: {dummy_image_with_table_path}")

        mm = ModelManager()
        if mm.get_yolo_model() is None:
            raise RuntimeError("YOLO Model could not be loaded. Check config and model path.")

        table_detector = TableDetector(model_manager=mm, crop_output_dir=dummy_crop_dir)
        cropped_tables = table_detector.detect_and_crop_tables(dummy_image_with_table_path)
        
        if cropped_tables:
            logger.info(f"Detected {len(cropped_tables)} tables. Cropped images saved in: {dummy_crop_dir}")
        else:
            logger.info(f"No tables detected or an error occurred for {dummy_image_with_table_path}.")

    except Exception as e:
        logger.error(f"Failed to run TableDetector example: {e}", exc_info=True)
    finally:
        if os.path.exists(dummy_image_with_table_path):
            os.remove(dummy_image_with_table_path)
        if os.path.exists(dummy_crop_dir):
            import shutil
            shutil.rmtree(dummy_crop_dir)
            logger.info(f"Cleaned up dummy crop directory: {dummy_crop_dir}") 