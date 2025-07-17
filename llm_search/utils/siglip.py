from transformers import AutoModel, AutoProcessor
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

    def __init__(self, model_name="google/siglip2-so400m-patch14-224", temperature=0.07):
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
        self.negative_prompts = ["a photo of nothing", "random background"]

    def compute_confidence(self, frame: np.ndarray, goal: list[str]) -> float:
        """
        Computes the cosine similarity between the image and the goal text using embeddings.

        Args:
            frame (np.ndarray): The input image as a NumPy array.
            goal (list[str]): The goal text to compare against the image.

        Returns:
            float: The cosine similarity score between image and text embeddings.
        """
        all_prompts = goal + self.negative_prompts

        # image = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        # outputs = self.pipe(image, candidate_labels=goal)

        # average_score = 0.0
        # for output in outputs:
        #     average_score += output['score']
        # average_score /= len(outputs)
        # return average_score

        similarity = self.get_image_embedding(frame) @ self.get_text_embedding(all_prompts).T

        print("Similarity", similarity.cpu().numpy())

        logits = similarity / self.temperature
        probs = torch.softmax(logits, dim=-1)
        # Take sum of probs for cat prompts vs sum for negative
        cat_probs = probs[:, :len(goal)].max().sum()
        neg_probs = probs[:, len(goal):].max().sum()

        final_score = cat_probs / (cat_probs + neg_probs)
        return final_score.item()  # Convert to Python float for easier handling

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
    image = Image.open("llm_search/utils/test_images/silver_cat.jpg").convert("RGB")
    image = np.array(image)
    goal = ["a photo of a cat"]
    score = siglip.compute_confidence(image, goal)
    print(f"Cosine similarity score: {score:.4f}")
    
if __name__ == "__main__":
    main()