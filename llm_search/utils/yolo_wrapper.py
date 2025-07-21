import ultralytics 
import torch
import numpy as np
import cv2

class YOLOWrapper:
    def __init__(self, model_path='yoloe-11m-seg.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ultralytics.YOLO(model_path)

    def set_classes(self, classes: list[str]):
        """
        Set the vocabulary for the YOLO-World model.
        """
        if hasattr(self.model, 'set_classes'):
            self.model.set_classes(classes, self.model.get_text_pe(classes)) # type: ignore

    def detect(self, image: np.ndarray):
        """
        Performs detection and segmentation, returning boxes, scores, classes, and masks.
        """
        results = self.model.predict(image, verbose=False)
        
        boxes = np.array([])
        scores = np.array([])
        class_ids = np.array([])
        masks = np.array([])

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy() # type: ignore
                scores = result.boxes.conf.cpu().numpy() # type: ignore
                class_ids = result.boxes.cls.cpu().numpy() # type: ignore

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy() # type: ignore

        return boxes, scores, class_ids, masks
    

def main():
    model = YOLOWrapper()
    model.set_classes(['chair', 'cup', 'telephone'])
    # Example usage with a dummy image
    dummy_image = cv2.imread('image.png')
    if dummy_image is None:
        print("Error: Could not read the image.")
        return
    dummy_image = cv2.resize(dummy_image, (640, 480))  # Resize to match model input size
    dummy_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
    
    boxes, scores, class_ids, masks = model.detect(dummy_image)
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Class IDs:", class_ids)
    print("Masks:", masks)

if __name__ == "__main__":
    main()