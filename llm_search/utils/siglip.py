from transformers import AutoProcessor, AutoModel
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers.pipelines import pipeline


class SigLipInterface:
    """
    SigLip is a class for computing cosine similarity between images and text using embeddings.
    It uses the Hugging Face Transformers library to load the model and extract embeddings.
    """

    def __init__(self, model_name="google/siglip-base-patch16-224", temperature=0.07):
        """
        Initializes the SigLip model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()  # Set to evaluation mode
        self.temperature = temperature  # Temperature scaling for cosine similarity
        self.pipe = pipeline(
            model=self.model_name,
            task="zero-shot-image-classification",
            device=self.device,
        )

    def compute_confidence(self, frame: np.ndarray, goal: list[str]) -> float:
        """
        Computes the cosine similarity between the image and the goal text using embeddings.

        Args:
            frame (np.ndarray): The input image as a NumPy array.
            goal (list[str]): The goal text to compare against the image.

        Returns:
            float: The cosine similarity score between image and text embeddings.
        """

        image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        outputs = self.pipe(image, candidate_labels=goal)

        average_score = 0.0
        for output in outputs:
            average_score += output['score']
        average_score /= len(outputs)
        return average_score
        
        # similarity = (self.get_image_embedding(frame) @ self.get_text_embedding(goal).T).squeeze(0) 
        # final_score = similarity.mean().item()
        # return final_score

    def get_image_embedding(self, frame: np.ndarray) -> torch.Tensor:
        """
        Get the image embedding for the given frame.

        Args:
            frame (np.ndarray): The input image as a NumPy array.
 
        Returns:
            torch.Tensor: The normalized image embedding.
        """
        image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        
        inputs = self.processor(
            images=[image], 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)
        embeds = F.normalize(embeds, dim=-1)
        return embeds
    
    def get_text_embedding(self, prompts: list[str]) -> torch.Tensor:
        """
        Get the text embedding for the given text.
        
        Args:
            prompts (list[str]): The input text as a list of strings.
            
        Returns:
            torch.Tensor: The normalized text embedding.
        """

        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            embeds = self.model.get_text_features(**inputs)
        embeds = F.normalize(embeds, dim=-1)
        return embeds
    


def main():
    # Example usage
    siglip = SigLipInterface()
    print(torch.cuda.is_available())
    image = Image.open("llm_search/utils/living_room.jpg").convert("RGB")
    image = np.array(image)
    goal = ["an image of a silver cat lying on wooden floor",
            "a photo of a cat", "a close-up photo of a cat",
            "a photo in the direction of a silver cat",
            ]

    score = siglip.compute_confidence(image, goal)
    print(f"Cosine similarity score: {score:.4f}")

if __name__ == "__main__":
    main()