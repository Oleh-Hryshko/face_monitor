import cv2
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import locale


def collect_image_files(input_dir, image_extensions):
    """Recursively collect image files from input_dir, including nested folders."""
    files = []
    input_path = Path(input_dir)
    for root, _, filenames in os.walk(input_path):
        root_path = Path(root)
        for filename in filenames:
            file_path = root_path / filename
            if file_path.suffix.lower() in image_extensions:
                files.append(file_path)
    files.sort()
    return files

def safe_read_image(file_path):
    """
    Safe image reading with Cyrillic path support.
    """
    try:
        # Try reading via numpy byte buffer to support Unicode paths on Windows.
        with open(file_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
    except Exception:
        return None

def safe_write_image(file_path, img):
    """
    Safe image saving with Cyrillic path support.
    """
    try:
        # Extract the file extension.
        ext = os.path.splitext(file_path)[1].lower()
        
        # Encode image before writing bytes to disk.
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(file_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception:
        return False

def remove_background_grabcut(img, face_region=None):
    """
    Remove background using the GrabCut algorithm.

    Parameters:
    - img: source image
    - face_region: face area (x, y, w, h) to improve segmentation

    Returns:
    - image with transparent background (RGBA)
    """
    height, width = img.shape[:2]
    
    # Create mask for GrabCut.
    mask = np.zeros((height, width), np.uint8)
    
    # Initialize rectangle for foreground extraction.
    if face_region:
        x, y, w, h = face_region
        # Expand area for better face capture.
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        rect = (max(0, x - pad_x), max(0, y - pad_y), 
                min(width, w + pad_x*2), min(height, h + pad_y*2))
    else:
        # If no face region is provided, use a centered area.
        center_x, center_y = width // 2, height // 2
        rect_size = min(width, height) // 3
        rect = (center_x - rect_size, center_y - rect_size, 
                rect_size * 2, rect_size * 2)
    
    # Run GrabCut.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask (0 - background, 1 - foreground).
    mask_final = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    
    # Blur mask for smoother edges.
    mask_final = cv2.GaussianBlur(mask_final, (5, 5), 0)
    
    # Create RGBA image.
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask_final])
    
    return rgba, mask_final

def remove_background_contour(img, face_region=None):
    """
    Alternative method: contour-based background removal.
    """
    height, width = img.shape[:2]
    
    # Convert to HSV for better skin segmentation.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Skin color ranges in HSV.
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create skin mask.
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological operations to clean up the mask.
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # If face region is known, use it to improve the mask.
    if face_region:
        x, y, w, h = face_region
        face_mask = np.zeros((height, width), np.uint8)
        face_mask[y:y+h, x:x+w] = 255
        
        # Combine masks.
        skin_mask = cv2.bitwise_and(skin_mask, face_mask)
    
    # Find contours.
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Use the largest contour.
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Build mask from contour.
        contour_mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        # Blur mask.
        contour_mask = cv2.GaussianBlur(contour_mask, (5, 5), 0)
        
        # Create RGBA image.
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, contour_mask])
        
        return rgba, contour_mask
    
    return None, None

def process_photos(input_dir, output_dir, padding=0.3, target_size=(224, 224), 
                   save_debug=False, preserve_aspect=True, remove_bg=False, 
                   bg_method="grabcut", bg_color=(255, 255, 255)):
    """
    Process all photos in the given directory:
    - Detect faces
    - Crop around face region
    - Optionally remove background
    - Save results while preserving aspect ratio

    Parameters:
    - input_dir: source photo directory
    - output_dir: output directory
    - padding: face margin (0.3 = 30%)
    - target_size: target output size
    - save_debug: whether to save debug images
    - preserve_aspect: whether to preserve face aspect ratio
    - remove_bg: whether to remove background
    - bg_method: background removal method ("grabcut" or "contour")
    - bg_color: replacement background color (RGB)
    """
    
    # Validate input directory.
    if not os.path.exists(input_dir):
        print(f"ERROR: Directory {input_dir} does not exist!")
        return False
    
    # Create output directory.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create directory for debug images.
    if save_debug:
        debug_dir = os.path.join(output_dir, "debug")
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
    
    # Create directory for masks.
    if remove_bg and save_debug:
        masks_dir = os.path.join(output_dir, "masks")
        Path(masks_dir).mkdir(parents=True, exist_ok=True)
    
    # Load classifiers.
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )
    
    # Supported image formats.
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Processing statistics.
    total = 0
    processed = 0
    no_face = 0
    errors = 0
    stretched = 0
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Padding: {padding*100:.0f}%")
    print(f"Aspect ratio mode: {'Yes' if preserve_aspect else 'No (stretching)'}")
    print(f"Background removal: {'Yes' if remove_bg else 'No'}")
    if remove_bg:
        print(f"Background removal method: {bg_method}")
        print(f"Background color: RGB{bg_color}")
    print("-" * 60)
    
    # Build recursive file list with Cyrillic path support.
    files = collect_image_files(input_dir, image_extensions)
    
    if not files:
        print("Input directory contains no images!")
        return False
    
    print(f"Images found: {len(files)}")
    print("-" * 60)
    
    # Process files.
    input_base = Path(input_dir).resolve()
    output_base = Path(output_dir)

    for idx, file_path in enumerate(files, 1):
        file_path = Path(file_path)
        rel_path = file_path.resolve().relative_to(input_base)
        rel_stem = rel_path.with_suffix("")
        
        total += 1
        print(f"[{idx}/{len(files)}] Processing: {rel_path}...")
        
        try:
            # Read image with Cyrillic path support.
            img = safe_read_image(file_path)
            if img is None:
                print(f"  ❌ Failed to load image")
                errors += 1
                continue
            
            # Read image dimensions.
            height, width = img.shape[:2]
            
            # Convert to grayscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces.
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # If no faces found, try profile detector.
            if len(faces) == 0:
                faces = profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            
            if len(faces) == 0:
                print(f"  ⚠️ Face not detected")
                no_face += 1
                continue
            
            # Pick the largest face.
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Remove background if requested.
            if remove_bg:
                print(f"  🎨 Removing background...")
                
                if bg_method == "grabcut":
                    rgba_img, mask = remove_background_grabcut(img, (x, y, w, h))
                else:  # contour
                    rgba_img, mask = remove_background_contour(img, (x, y, w, h))
                
                if rgba_img is not None:
                    # Create a new background.
                    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])  # BGR order for OpenCV
                    new_bg = np.full_like(img, bg_color_bgr)
                    
                    # Normalize mask.
                    mask_normalized = mask.astype(np.float32) / 255.0
                    
                    # Composite foreground onto the new background.
                    for c in range(3):
                        new_bg[:, :, c] = (new_bg[:, :, c] * (1 - mask_normalized) + 
                                          rgba_img[:, :, c] * mask_normalized)
                    
                    img = new_bg.astype(np.uint8)

                    if save_debug:
                        # Save mask with Cyrillic path support.
                        mask_path = output_base / "masks" / rel_stem.parent / f"mask_{rel_stem.name}.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        safe_write_image(mask_path, mask)
                        print(f"  💾 Mask saved")
                else:
                    print(f"  ⚠️ Failed to remove background, using original image")
            
            # Add padding around face area.
            padding_x = int(w * padding)
            padding_y = int(h * padding)
            
            # Calculate crop bounds.
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + w + padding_x)
            y2 = min(height, y + h + padding_y)
            
            # Crop face area.
            cropped_face = img[y1:y2, x1:x2]
            crop_height, crop_width = cropped_face.shape[:2]
            
            print(f"  Face area: {crop_width}x{crop_height}")
            
            # Preserve aspect ratio.
            if preserve_aspect:
                if target_size:
                    scale = min(target_size[0] / crop_width, target_size[1] / crop_height)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    
                    resized = cv2.resize(cropped_face, (new_width, new_height), 
                                        interpolation=cv2.INTER_LANCZOS4)
                    
                    canvas = np.full((target_size[1], target_size[0], 3), 
                                   bg_color[2], dtype=np.uint8) if remove_bg else \
                            np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    
                    x_offset = (target_size[0] - new_width) // 2
                    y_offset = (target_size[1] - new_height) // 2
                    
                    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                    final_face = canvas
                    print(f"  Resized: {new_width}x{new_height} -> centered on canvas")
                else:
                    final_face = cropped_face
            else:
                if target_size:
                    final_face = cv2.resize(cropped_face, target_size, 
                                           interpolation=cv2.INTER_LANCZOS4)
                    stretched += 1
                else:
                    final_face = cropped_face
            
            # Save result with Cyrillic path support.
            ext = ".png" if remove_bg else ".jpg"
            output_path = output_base / rel_stem.parent / f"{rel_stem.name}{ext}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_write_image(output_path, final_face)
            
            print(f"  ✅ Saved: {output_path.relative_to(output_base)}")
            
            # Debug image.
            if save_debug:
                # Reload original image for debug rendering.
                img_with_box = safe_read_image(file_path)
                if img_with_box is not None:
                    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    
                    debug_path = output_base / "debug" / rel_path.parent / f"debug_{rel_path.name}"
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    safe_write_image(debug_path, img_with_box)
            
            processed += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            errors += 1
    
    # Summary.
    print("-" * 60)
    print("PROCESSING COMPLETE!")
    print(f"Total images: {total}")
    print(f"Successfully processed: {processed}")
    print(f"No face detected: {no_face}")
    print(f"Errors: {errors}")
    if stretched > 0:
        print(f"⚠️ Stretched images: {stretched}")
    print(f"Results saved to: {output_dir}")
    
    return processed > 0

def main():
    # Set console encoding.
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Background removal and face cropping tool')
    parser.add_argument('input_dir', help='Directory with source photos')
    parser.add_argument('-o', '--output', default=None, 
                       help='Directory for output files')
    parser.add_argument('-p', '--padding', type=float, default=0.3,
                       help='Padding around face (0.0-0.5, default: 0.3)')
    parser.add_argument('-s', '--size', type=int, nargs=2, default=[224, 224],
                       help='Output image size (width height, default: 224 224)')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Save debug images')
    parser.add_argument('--stretch', action='store_true',
                       help='Stretch faces (aspect ratio is preserved by default)')
    parser.add_argument('--no-resize', action='store_true',
                       help='Do not resize (keep original cropped region size)')
    
    # Background-removal options.
    parser.add_argument('--remove-bg', action='store_true',
                       help='Remove image background')
    parser.add_argument('--bg-method', choices=['grabcut', 'contour'], default='grabcut',
                       help='Background removal method (grabcut = more accurate, contour = faster)')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[255, 255, 255],
                       help='Replacement background color (R G B, default: 255 255 255 = white)')
    
    args = parser.parse_args()
    
    # Resolve output directory.
    if args.output is None:
        output_dir = os.path.join(args.input_dir, 'processed_faces')
    else:
        output_dir = args.output
    
    # Validate padding.
    if args.padding < 0 or args.padding > 1.0:
        print("Error: padding must be between 0 and 1.0")
        return
    
    # Resolve target size.
    target_size = None if args.no_resize else tuple(args.size)
    
    print("=" * 60)
    print("BACKGROUND REMOVAL AND FACE CROPPING")
    print("=" * 60)
    
    # Run processing.
    process_photos(
        input_dir=args.input_dir,
        output_dir=output_dir,
        padding=args.padding,
        target_size=target_size,
        save_debug=args.debug,
        preserve_aspect=not args.stretch,
        remove_bg=args.remove_bg,
        bg_method=args.bg_method,
        bg_color=tuple(args.bg_color)
    )

if __name__ == "__main__":
    main()