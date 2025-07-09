from transformers.pipelines import pipeline
import numpy as np
import torch
from PIL import Image

class SigLipInterface:
    """
    SigLip is a class for performing zero-shot image classification using the SigLip model.
    It uses the Hugging Face Transformers library to load the model and perform inference.
    """

    def __init__(self, model_name="google/siglip2-base-patch32-256"):
        """
        Initializes the SigLip model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pipe = pipeline(
            model=self.model_name,
            task="zero-shot-image-classification",
            device=self.device
        )

    def cosine_similarity(self, frame: np.ndarray, goal: str) -> float:
        """
        Computes the cosine similarity between the image and the goal.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            goal (str): The goal text to compare against the image.

        Returns:
            float: The cosine similarity score.
        """
        image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        inputs = {
            # "images": [image],
            "texts": [f"a picture of a {goal}", f"an image of a {goal}", f"a photo of a {goal}"]
        }
        outputs = self.pipe(image, candidate_labels=inputs["texts"])
        max_score = outputs[0]['score']
        return max_score