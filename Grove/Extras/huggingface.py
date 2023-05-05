import requests
import numpy as np
import time

class HuggingFaceEmbeddings:
    def __init__(self, tokens) -> None:
        self.tokens = tokens
        model = "sentence-transformers/all-mpnet-base-v2"
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/" + model
    
    def __call__(self, input: str) -> np.array:
        return self.embed(input)
    
    def embed(self, input: str, token_choice = 0) -> np.array:
        token_choice = token_choice % len(self.tokens)
        token = self.tokens[token_choice]
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": input,
            "wait_for_model": False,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        res = response.json()
        if isinstance(res, dict): # If the model is not ready yet
            time.sleep(1)
            return self.embed(input, token_choice = token_choice + 1)
        else:
            return np.array(res)


	