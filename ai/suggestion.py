import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class SuggestionModel:
    def __init__(self):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        # IMPORTANT: GPT2 doesn't have pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate_sync(self, text: str) -> str:
        # Convert text → tokens
        inputs = self.tokenizer(text, return_tensors="pt")

        # Length of input tokens
        input_length = inputs["input_ids"].shape[1]

        # Generate output (no gradient needed)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract only newly generated tokens
        generated_tokens = outputs[0][input_length:]

        # Decode only generated part
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return generated_text.strip()

    async def generate(self, text: str) -> str:
        # Run blocking code in separate thread (non-blocking API)
        return await asyncio.to_thread(self._generate_sync, text)






