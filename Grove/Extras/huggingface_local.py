from sentence_transformers import SentenceTransformer
import numpy as np

class HuggingFaceLocal:
    """
    A wrapper around the HuggingFace SentenceTransformer library that allows it to be used as an econder
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v1", device: str = "cpu") -> None:
        """
        Initializes a HuggingFaceLocal object

        Args:
            model_name: A string representing the name of the model to use
            device: A string representing the device to use (e.g. "cpu" or "cuda")
        """
        self.model = SentenceTransformer(model_name).to(device)
    
    def __call__(self, input: str) -> np.array:
        """
        Returns the embedding of the input string

        Args:
            input: A string representing the input to embed

        Returns:
            A numpy array representing the embedding of the input string
        """
        return self.embed(input)
    
    def embed(self, input: str) -> np.array:
        """
        Returns the embedding of the input string
        
        Args:
            input: A string representing the input to embed
        
        Returns:
            A numpy array representing the embedding of the input string
        """
        return np.array(self.model.encode(input))


	