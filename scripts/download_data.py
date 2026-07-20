import os
import json
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from datasets import load_dataset
from PIL import Image

def main():
    print("Loading ICDAR-2019-SROIE dataset from Hugging Face...")
    try:
        # Load the dataset
        dataset = load_dataset("jsdnrs/ICDAR2019-SROIE", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to another public mirror/version if possible...")
        raise e
        
    print(f"Dataset splits found: {list(dataset.keys())}")
    
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    
    for split in dataset.keys():
        print(f"\nProcessing split: {split}")
        split_dir = os.path.join(base_data_dir, split)
        image_dir = os.path.join(split_dir, "images")
        ann_dir = os.path.join(split_dir, "annotations")
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        subset = dataset[split]
        print(f"Saving {len(subset)} items to {split_dir}...")
        
        for idx, item in enumerate(tqdm(subset)):
            # Determine filename
            # Some SROIE datasets might store metadata/filename. If not, use index
            filename = f"receipt_{split}_{idx:04d}"
            
            # Save image
            img = item.get("image")
            if img is not None:
                # If image is not PIL, convert it
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                img.save(os.path.join(image_dir, f"{filename}.jpg"), "JPEG")
            
            # Save key information extraction (KIE) entities
            entities = item.get("entities")
            if entities is not None:
                ent_path = os.path.join(ann_dir, f"{filename}.json")
                with open(ent_path, "w", encoding="utf-8") as f:
                    json.dump(entities, f, indent=4, ensure_ascii=False)
            
            # Save text detection / OCR tokens (words and bboxes)
            words = item.get("words")
            bboxes = item.get("bboxes")
            if words is not None or bboxes is not None:
                ocr_path = os.path.join(ann_dir, f"{filename}.ocr.json")
                ocr_data = {
                    "words": words if words is not None else [],
                    "bboxes": bboxes if bboxes is not None else []
                }
                with open(ocr_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_data, f, indent=4, ensure_ascii=False)
                    
    print("\nDownload and structuring completed successfully!")
    print(f"Data saved to: {base_data_dir}")

if __name__ == "__main__":
    main()
