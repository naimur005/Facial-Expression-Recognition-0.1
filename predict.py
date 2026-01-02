"""
Use a trained model to predict emotions from images.
Usage:
    python predict.py                     
    python predict.py image.jpg           
    python predict.py img1.jpg img2.jpg   
    python predict.py --folder Pictures/  

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


MODEL_PATH = './fer_model_final.keras'  
IMG_SIZE = (48, 48)                     
CLASS_LABELS = ['Disappointed', 'Happy']


def load_fer_model(model_path=MODEL_PATH):
    """Load the trained model."""
    if not os.path.exists(model_path):
        alternatives = [
            './fer_model_best.keras',
            './fer_model_final.h5',
            './fer_model_best.h5',
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                break
        else:
            print(f"❌ Model not found at: {model_path}")
            print("   Please train the model first using: python train.py")
            sys.exit(1)
    
    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    return model


def predict_single(model, img_path, show=True):
    """
    Predict emotion from a single image.
    
    Args:
        model: Trained Keras model
        img_path: Path to image file
        show: Whether to display the result
    
    Returns:
        dict with label, confidence, and probabilities
    """
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None
    
    from PIL import Image
    img_pil = Image.open(img_path).convert('L').resize(IMG_SIZE) 
    img_array = np.array(img_pil, dtype=np.float32)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=-1)  
    img_batch = np.expand_dims(img_array, axis=0)   
    
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    img_display = image.load_img(img_path)
    
    result = {
        'path': img_path,
        'label': CLASS_LABELS[predicted_class],
        'confidence': float(confidence),
        'probabilities': {
            CLASS_LABELS[i]: float(predictions[0][i])
            for i in range(len(CLASS_LABELS))
        }
    }
    
    if show:
        display_result(img_display, result, predictions[0])
    
    return result


def display_result(img, result, probabilities):
    """Display image with prediction results - compact size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    ax1.imshow(img)
    color = 'green' if result['label'] == 'Happy' else 'darkorange'
    ax1.set_title(
        f"Predicted: {result['label']}\nConfidence: {result['confidence']:.1%}",
        fontsize=16, fontweight='bold', color=color
    )
    ax1.axis('off')
    
    colors = ['#ff7f0e', '#2ca02c']
    bars = ax2.barh(CLASS_LABELS, probabilities, color=colors, height=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14)
    
    for bar, val in zip(bars, probabilities):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.1%}', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def predict_folder(model, folder_path, show_each=False):
    """Predict emotions for all images in a folder."""
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    image_files = []
    for f in os.listdir(folder_path):
        if os.path.splitext(f)[1].lower() in valid_extensions:
            image_files.append(os.path.join(folder_path, f))
    
    if not image_files:
        print(f"❌ No images found in: {folder_path}")
        return []
    
    print(f"\n📁 Found {len(image_files)} images in {folder_path}")
    print("-" * 50)
    
    results = []
    happy_count = 0
    disappointed_count = 0
    
    for i, img_path in enumerate(image_files):
        result = predict_single(model, img_path, show=show_each)
        if result:
            results.append(result)
            
            if result['label'] == 'Happy':
                happy_count += 1
            else:
                disappointed_count += 1
            
            emoji = '😊' if result['label'] == 'Happy' else '😞'
            print(f"[{i+1}/{len(image_files)}] {os.path.basename(img_path)}: "
                  f"{emoji} {result['label']} ({result['confidence']:.1%})")
    
    print("-" * 50)
    print(f"\n📊 Summary:")
    print(f"   Happy: {happy_count} ({happy_count/len(results)*100:.1f}%)")
    print(f"   Disappointed: {disappointed_count} ({disappointed_count/len(results)*100:.1f}%)")
    
    return results


def interactive_mode(model):
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("🎭 Interactive Prediction Mode")
    print("="*60)
    print("Enter image path to predict (or 'q' to quit)")
    print("You can also enter a folder path to predict all images")
    print()
    
    while True:
        try:
            path = input("📷 Image/Folder path: ").strip()
            
            if path.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not path:
                continue
            
            if os.path.isdir(path):
                predict_folder(model, path, show_each=True)
            elif os.path.isfile(path):
                result = predict_single(model, path, show=True)
                if result:
                    print(f"\n   Result: {result['label']} "
                          f"(Confidence: {result['confidence']:.1%})")
            else:
                print(f"❌ Path not found: {path}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function."""
    print("\n" + "="*60)
    print("🎭 Facial Expression Recognition - Prediction")
    print("="*60)
    
    model = load_fer_model()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--folder' and len(sys.argv) > 2:
            predict_folder(model, sys.argv[2], show_each=True)
        else:
            for img_path in sys.argv[1:]:
                if os.path.isdir(img_path):
                    predict_folder(model, img_path, show_each=True)
                else:
                    print(f"\n📷 Processing: {img_path}")
                    result = predict_single(model, img_path, show=True)
                    if result:
                        print(f"   Result: {result['label']} "
                              f"(Confidence: {result['confidence']:.1%})")
    else:
        interactive_mode(model)

if __name__ == "__main__":
    main()
