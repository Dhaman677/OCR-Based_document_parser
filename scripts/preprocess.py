import os
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

class ReceiptPreprocessor:
    """
    Image preprocessing pipeline for scanned receipts and invoices.
    Includes deskewing, denoising, contrast enhancement, and binarization.
    """
    def __init__(self, 
                 binarize_method: str = "adaptive", 
                 adaptive_block_size: int = 15, 
                 adaptive_c: int = 10,
                 denoise_d: int = 5,
                 denoise_sigma_color: int = 75,
                 denoise_sigma_space: int = 75):
        self.binarize_method = binarize_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.denoise_d = denoise_d
        self.denoise_sigma_color = denoise_sigma_color
        self.denoise_sigma_space = denoise_sigma_space

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detects the skew angle of text lines in the receipt image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert binarization (white text on black background)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Dilate horizontally to merge letters/words into text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find all contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            
            # Filter out small noise contours
            if w < 40 or h < 10:
                continue
                
            angle = rect[2]
            
            # OpenCV's cv2.minAreaRect angle calculation depends on width vs height
            if w < h:
                angle = angle + 90
                
            # Normalize angle to range [-45, 45]
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
                
            angles.append(angle)
            
        if not angles:
            return 0.0
            
        # Return median angle to ignore outliers (e.g. non-text vertical lines, borders)
        return float(np.median(angles))

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates the image by the given angle around its center, adding white borders.
        """
        if abs(angle) < 0.2:  # Ignore negligible skew to save compute and prevent interpolation blur
            return image.copy()
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Compute new bounding dimensions to avoid cropping corners
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix translation component
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Warp image
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
        return rotated

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Bilateral Filtering to smooth out background noise while preserving edges.
        """
        return cv2.bilateralFilter(
            image, 
            d=self.denoise_d, 
            sigmaColor=self.denoise_sigma_color, 
            sigmaSpace=self.denoise_sigma_space
        )

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhances local contrast using CLAHE.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def binarize(self, gray: np.ndarray, method: str = None) -> np.ndarray:
        """
        Converts grayscale image to binary black-and-white.
        """
        method = method or self.binarize_method
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=self.adaptive_block_size, 
                C=self.adaptive_c
            )
        else:
            raise ValueError(f"Unknown binarization method: {method}")
        return binary

    def preprocess_image(self, image: np.ndarray) -> dict:
        """
        Executes the full preprocessing pipeline on the image.
        Returns a dict of all intermediate steps and the final binary image.
        """
        # 1. Deskew
        skew_angle = self.detect_skew_angle(image)
        deskewed = self.rotate_image(image, skew_angle)
        
        # 2. Denoise
        denoised = self.denoise(deskewed)
        
        # 3. Grayscale conversion
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 4. Contrast Enhancement
        enhanced = self.enhance_contrast(gray)
        
        # 5. Binarization
        binary = self.binarize(enhanced)
        
        return {
            "original": image,
            "skew_angle": skew_angle,
            "deskewed": deskewed,
            "denoised": denoised,
            "gray": gray,
            "enhanced": enhanced,
            "binary": binary
        }

def preprocess_single_image(args):
    """
    Worker function to process a single image.
    args: tuple of (src_path, dst_path, preprocessor_params)
    """
    src_path, dst_path, params = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False, src_path
        preprocessor = ReceiptPreprocessor(**params)
        res = preprocessor.preprocess_image(img)
        cv2.imwrite(dst_path, res["binary"])
        return True, src_path
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False, src_path

def process_dataset(input_dir: str, output_dir: str, preprocessor: ReceiptPreprocessor):
    """
    Finds all image directories inside input_dir, preprocesses images,
    and writes them under the identical folder structure in output_dir
    using parallel processes.
    """
    print(f"Starting batch preprocessing from '{input_dir}' to '{output_dir}'...")
    
    preprocessor_params = {
        "binarize_method": preprocessor.binarize_method,
        "adaptive_block_size": preprocessor.adaptive_block_size,
        "adaptive_c": preprocessor.adaptive_c,
        "denoise_d": preprocessor.denoise_d,
        "denoise_sigma_color": preprocessor.denoise_sigma_color,
        "denoise_sigma_space": preprocessor.denoise_sigma_space
    }
    
    tasks = []
    for root, dirs, files in os.walk(input_dir):
        images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not images:
            continue
            
        rel_path = os.path.relpath(root, input_dir)
        target_img_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_img_dir, exist_ok=True)
        
        for img_name in images:
            src_path = os.path.join(root, img_name)
            dst_path = os.path.join(target_img_dir, img_name)
            tasks.append((src_path, dst_path, preprocessor_params))
            
    print(f"Total images to process: {len(tasks)}")
    
    count = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(preprocess_single_image, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            success, path = future.result()
            if success:
                count += 1
            else:
                print(f"Failed to process image: {path}")
                
    print(f"Batch preprocessing completed! Successfully processed {count}/{len(tasks)} images.")

def generate_visualizations(input_dir: str, viz_dir: str, preprocessor: ReceiptPreprocessor, num_samples: int = 5):
    """
    Selects random samples and saves side-by-side comparison images.
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    # Gather all images in the input directory tree
    all_image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_image_paths.append(os.path.join(root, f))
                
    if not all_image_paths:
        print("No images found for creating visualizations.")
        return
        
    random.seed(42)  # For reproducible samples
    samples = random.sample(all_image_paths, min(num_samples, len(all_image_paths)))
    
    print(f"Generating visual comparisons for {len(samples)} sample images in '{viz_dir}'...")
    
    for idx, path in enumerate(samples):
        img_name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Get intermediate stages
        stages = preprocessor.preprocess_image(img)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 7))
        
        # 1. Original
        axes[0].imshow(cv2.cvtColor(stages["original"], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original: {img_name}")
        axes[0].axis("off")
        
        # 2. Deskewed
        axes[1].imshow(cv2.cvtColor(stages["deskewed"], cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Deskewed (Angle: {stages['skew_angle']:.2f}°)")
        axes[1].axis("off")
        
        # 3. Contrast Enhanced (Grayscale)
        axes[2].imshow(stages["enhanced"], cmap="gray")
        axes[2].set_title("Denoised + CLAHE")
        axes[2].axis("off")
        
        # 4. Final Binary
        axes[3].imshow(stages["binary"], cmap="gray")
        axes[3].set_title(f"Binary ({preprocessor.binarize_method})")
        axes[3].axis("off")
        
        plt.tight_layout()
        output_path = os.path.join(viz_dir, f"comparison_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
    print("Visual comparisons generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Image Preprocessing Tool")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Path to raw images folder")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to processed output folder")
    parser.add_argument("--viz_dir", type=str, default="data/processed/visualizations", help="Path to save comparison charts")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--binarize_method", type=str, choices=["adaptive", "otsu"], default="adaptive", help="Binarization algorithm")
    parser.add_argument("--block_size", type=int, default=15, help="Adaptive threshold block size")
    parser.add_argument("--c_val", type=int, default=10, help="Adaptive threshold C constant")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ReceiptPreprocessor(
        binarize_method=args.binarize_method,
        adaptive_block_size=args.block_size,
        adaptive_c=args.c_val
    )
    
    # 1. Generate visual comparisons
    generate_visualizations(args.input_dir, args.viz_dir, preprocessor, args.num_samples)
    
    # 2. Batch process entire dataset
    process_dataset(args.input_dir, args.output_dir, preprocessor)
