# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2Model
import torch.nn.functional as F


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_name)  # type: ignore
        self.model = Blip2Model.from_pretrained(model_name).to(device)
        self.model.eval()

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        
        # Process the image and text using the Hugging Face processor
        inputs = self.processor(  # type: ignore
            images=pil_img, 
            text=txt, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Get the image and text embeddings
            outputs = self.model(**inputs)
            
            # Extract image and text features
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize the embeddings
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            # Compute cosine similarity
            cosine = torch.sum(image_embeds * text_embeds, dim=-1).item()

        return cosine

def main():
    # Example usage
    model = BLIP2ITM()
    image = Image.open("llm_search/utils/test_images/silver_cat.jpg").convert("RGB")
    image = np.array(image)
    text = "a photo of a cat"
    score = model.cosine(image, text)
    print(f"Cosine similarity score: {score}")

if __name__ == "__main__":
    main()
