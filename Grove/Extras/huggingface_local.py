from sentence_transformers import SentenceTransformer
import numpy as np

class HuggingFaceLocal:
    def __init__(self) -> None:
        model_name = "sentence-transformers/all-mpnet-base-v1"
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: str) -> np.array:
        return self.embed(input)
    
    def embed(self, input: str, token_choice = 0) -> np.array:
        return np.array(self.model.encode(input))


	