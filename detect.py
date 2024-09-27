from ultralytics import YOLO

model = YOLO("weights/best.pt")

def detect_objects():
    results = model("screenshot.png")

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename="result.jpg")  # save to disk
