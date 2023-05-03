import requests
import numpy as np
import time


class HuggingFaceEmbeddings:
    def __init__(self, token) -> None:
        self.token = token
        model = "sentence-transformers/all-mpnet-base-v2"
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/" + model
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def __call__(self, input: str) -> np.array:
        payload = {
            "inputs": input,
            "wait_for_model": True,
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        res = response.json()
        if isinstance(res, dict): # If the model is not ready yet
            time.sleep(5)
            return self.__call__(input)
        else:
            return np.array(res)


	