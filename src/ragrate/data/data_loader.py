# data_loader.py

import json
from typing import Dict, List, Tuple

class DataLoader:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.prompts = []
        self.answers = []

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load prompts and ground truth answers from a JSON file.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists, one for prompts and one for answers.
        """
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        for item in data:
            prompt = item['prompt']
            answer = item['answer']
            self.prompts.append(prompt)
            self.answers.append(answer)

        return self.prompts, self.answers

    def get_data(self) -> Dict[str, List[str]]:
        """
        Get the loaded prompts and answers as a dictionary.

        Returns:
            Dict[str, List[str]]: A dictionary containing prompts and answers.
        """
        return {'prompts': self.prompts, 'answers': self.answers}