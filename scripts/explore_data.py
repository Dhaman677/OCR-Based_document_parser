import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

def main():
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    raw_dir = os.path.join(base_data_dir, "raw")
    
    print("--- SROIE Dataset Exploration ---")
    
    # 1. Dataset statistics
    for split in ["train", "test"]:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory {split_dir} does not exist.")
            continue
            
        img_dir = os.path.join(split_dir, "images")
        ann_dir = os.path.join(split_dir, "annotations")
        
        images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        anns = [f for f in os.listdir(ann_dir) if f.endswith(".json") and not f.endswith(".ocr.json")]
        ocrs = [f for f in os.listdir(ann_dir) if f.endswith(".ocr.json")]
        
        print(f"Split '{split}':")
        print(f"  - Number of Images: {len(images)}")
        print(f"  - Number of KIE Annotations: {len(anns)}")
        print(f"  - Number of OCR Annotations: {len(ocrs)}")
        
    # 2. Inspect a sample KIE and OCR annotation
    train_ann_dir = os.path.join(raw_dir, "train", "annotations")
    train_img_dir = os.path.join(raw_dir, "train", "images")
    
    all_kie_files = [f for f in os.listdir(train_ann_dir) if f.endswith(".json") and not f.endswith(".ocr.json")]
    
    if len(all_kie_files) == 0:
        print("No KIE annotations found in train set.")
        return
        
    # Pick a random sample
    random.seed(42)  # For reproducibility
    sample_file = random.choice(all_kie_files)
    sample_id = os.path.splitext(sample_file)[0]
    
    kie_path = os.path.join(train_ann_dir, sample_file)
    ocr_path = os.path.join(train_ann_dir, f"{sample_id}.ocr.json")
    img_path = os.path.join(train_img_dir, f"{sample_id}.jpg")
    
    print(f"\nSample ID: {sample_id}")
    
    # Load KIE
    with open(kie_path, "r", encoding="utf-8") as f:
        kie_data = json.load(f)
    print("\n--- Key Information Extraction (KIE) ---")
    print(json.dumps(kie_data, indent=4))
    
    # Load OCR
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    words = ocr_data.get("words", [])
    bboxes = ocr_data.get("bboxes", [])
    print(f"\nOCR Word Count: {len(words)}")
    
    # 3. Create visualization
    if os.path.exists(img_path):
        print(f"\nGenerating visualization for {sample_id}.jpg...")
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, or fallback to default
        try:
            # Using basic default font (usually very small but always works)
            font = ImageFont.load_default()
        except:
            font = None
            
        # Draw bounding boxes
        for word, bbox in zip(words, bboxes):
            # bbox is [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            
            # Draw green rectangle for words
            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)
            
            # Optional: draw small text (in red) just above or inside box
            if font:
                draw.text((xmin, max(0, ymin - 12)), word, fill="red", font=font)
                
        # Save visualization to data directory
        output_path = os.path.join(base_data_dir, "sample_exploration.png")
        img.save(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        print(f"Image not found for visualization at: {img_path}")

if __name__ == "__main__":
    main()
