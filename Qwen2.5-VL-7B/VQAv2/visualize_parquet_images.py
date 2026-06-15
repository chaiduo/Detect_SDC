import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

PARQUET_PATH = "/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/validation-00000-of-00068.parquet"
OUTPUT_DIR = "/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/visualized_images2"
NUM_IMAGES = 100

def load_parquet(path):
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    return df

def decode_image(image_data):
    if isinstance(image_data, dict):
        if 'bytes' in image_data:
            return Image.open(io.BytesIO(image_data['bytes']))
        elif 'path' in image_data and os.path.exists(image_data['path']):
            return Image.open(image_data['path'])
    elif isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    return None

def visualize_images(df, num_images=16):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    displayed = 0
    for idx in range(min(num_images, len(df))):
        row = df.iloc[idx]
        img_data = row['image']
        img = decode_image(img_data)
        
        if img is not None:
            ax = axes[displayed]
            ax.imshow(img)
            
            question = row['question']
            answer = row['multiple_choice_answer']
            
            if len(question) > 50:
                question = question[:50] + "..."
            if len(answer) > 30:
                answer = answer[:30] + "..."
            
            ax.set_title(f"Q: {question}\nA: {answer}", fontsize=8)
            ax.axis('off')
            displayed += 1
        else:
            print(f"Warning: Could not decode image at index {idx}")
    
    for i in range(displayed, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "vqav2_validation_images.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()

def save_individual_images(df, num_images=16):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    saved = 0
    for idx in range(min(num_images, len(df))):
        row = df.iloc[idx]
        img_data = row['image']
        img = decode_image(img_data)
        
        if img is not None:
            question = row['question'][:30].replace(" ", "_")
            answer = row['multiple_choice_answer'][:20].replace(" ", "_")
            filename = f"img_{idx:04d}_Q_{question}_A_{answer}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            img.convert('RGB').save(filepath, 'JPEG', quality=95)
            saved += 1
            print(f"Saved: {filename}")
    
    print(f"Total saved: {saved} images")

def main():
    df = load_parquet(PARQUET_PATH)
    
    print("\nSample data:")
    print(df.head(2))
    
    print(f"\nDisplaying {NUM_IMAGES} images in grid...")
    visualize_images(df, NUM_IMAGES)
    
    print(f"\nSaving {NUM_IMAGES} individual images...")
    save_individual_images(df, NUM_IMAGES)

if __name__ == "__main__":
    main()
