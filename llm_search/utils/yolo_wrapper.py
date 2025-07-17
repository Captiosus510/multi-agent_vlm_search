import ultralytics

class YOLOWrapper:
    def __init__(self, model_path='yolo11m.pt'):
        self.model = ultralytics.YOLO(model_path)
        self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference

    def detect(self, image):
        results = self.model(image)
        return results