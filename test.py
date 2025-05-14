from ultralytics import YOLO
import sys
import os
import cv2
import numpy as np
from datetime import datetime
import argparse

def detect_elements(model_path, screenshot_path, confidence=0.01, iou=0.4, max_det=2048):
    """Run detection on the screenshot and return the results"""
    model = YOLO(model_path)
    results = model(screenshot_path, conf=confidence, max_det=max_det, iou=iou)
    return results[0]  # Return the first result

def filter_overlapping_elements(boxes, overlap_threshold=0.2):
    """Filter out overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    # Convert boxes to a format for processing
    formatted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        score = box[4] if len(box) > 4 else 1.0
        formatted_boxes.append([x1, y1, x2, y2, score])
    
    # Sort by area (smaller to larger)
    sorted_indices = sorted(range(len(formatted_boxes)), 
                           key=lambda i: (formatted_boxes[i][2] - formatted_boxes[i][0]) * 
                                        (formatted_boxes[i][3] - formatted_boxes[i][1]))
    
    keep_indices = []
    for i in sorted_indices:
        box_i = formatted_boxes[i]
        
        # Check if box_i significantly overlaps with any kept box
        should_keep = True
        for j in keep_indices:
            box_j = formatted_boxes[j]
            
            # Calculate intersection area
            x1 = max(box_i[0], box_j[0])
            y1 = max(box_i[1], box_j[1])
            x2 = min(box_i[2], box_j[2])
            y2 = min(box_i[3], box_j[3])
            
            if x1 < x2 and y1 < y2:
                intersection_area = (x2 - x1) * (y2 - y1)
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                smaller_area = min(area_i, area_j)
                
                if intersection_area / smaller_area > overlap_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep_indices.append(i)
    
    return [boxes[i] for i in keep_indices]

def draw_boxes_on_image(image_path, boxes, output_path=None, draw_labels=False, classes=None):
    """Draw bounding boxes on the image and save it"""
    image = cv2.imread(image_path)
    
    # Draw each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        
        # Draw rectangle with cyan color (BGR format)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Add semi-transparent fill
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), -1)  # Filled rectangle
        alpha = 0.2  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw label text if requested
        if draw_labels:
            # Get the class label if available
            if len(box) > 5 and classes is not None:
                class_idx = int(box[5])
                if class_idx < len(classes):
                    label = str(classes[class_idx])
                else:
                    label = f"Class {class_idx}"
            else:
                label = f"Element {i+1}"
            
            # Calculate label position (bottom left of box)
            text_position = (x1, y2 - 5)  # 5 pixels above the bottom left corner
            
            # Add background rectangle for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, 
                         (text_position[0] - 1, text_position[1] - text_size[1] - 5),
                         (text_position[0] + text_size[0] + 1, text_position[1] + 1),
                         (0, 0, 0), -1)
            
            # Draw the text
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'detected_elements_{timestamp}.png'
    
    cv2.imwrite(output_path, image)
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect UI elements in a screenshot')
    parser.add_argument('screenshot', help='Path to the screenshot image')
    parser.add_argument('--model', default="weights/cloud.pt", help='Path to YOLO model file')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('--max-det', type=int, default=2048, help='Maximum detections')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap threshold for filtering')
    parser.add_argument('--output', help='Output path (default: detected_elements_[timestamp].png)')
    parser.add_argument('--draw-labels', action='store_true', help='Draw label text on bottom left of each box')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.screenshot):
        print(f"Error: Screenshot file {args.screenshot} not found.")
        return
        
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    
    # Run detection
    print(f"Running element detection on {args.screenshot}...")
    results = detect_elements(args.model, args.screenshot, args.conf, args.iou, args.max_det)
    
    # Extract boxes and class information
    boxes = results.boxes.xyxy.cpu().numpy()
    if len(results.boxes) > 0 and hasattr(results.boxes, 'cls'):
        # Add class index to boxes
        class_indices = results.boxes.cls.cpu().numpy()
        boxes = np.column_stack((boxes, class_indices))
    
    # Get class names if available
    class_names = results.names if hasattr(results, 'names') else None
    
    # Print class names if available and draw-labels is enabled
    if args.draw_labels and class_names:
        print(f"Available classes: {class_names}")
    
    # Filter overlapping boxes
    print(f"Found {len(boxes)} elements, filtering overlaps...")
    filtered_boxes = filter_overlapping_elements(boxes, args.overlap)
    print(f"Kept {len(filtered_boxes)} elements after filtering")
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'detected_elements_{timestamp}.png'
    
    # Draw boxes on image
    output_path = draw_boxes_on_image(args.screenshot, filtered_boxes, output_path, args.draw_labels, class_names)
    print(f"Detection complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
