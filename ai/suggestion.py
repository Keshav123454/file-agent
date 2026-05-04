import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SuggestionModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def _generate_sync(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.7,
                top_k=50,
                do_sample=True
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(text):].strip()

    async def generate(self, text: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_sync, text)




