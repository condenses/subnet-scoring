from fastapi import FastAPI
from openai import OpenAI
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import uvicorn
from loguru import logger
from .schemas import ScoringRequest, ScoringResponse


class App:
    def __init__(self):
        self.app = FastAPI()
        self.llm_client = OpenAI()
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_base_reward = 0.75

        self.app.add_api_route(
            "/api/scoring",
            self.api_scoring,
            methods=["POST"],
            response_model=ScoringResponse,
        )

    @torch.no_grad()
    def api_scoring(self, request: ScoringRequest) -> ScoringResponse:
        original_tokenized = self.reward_tokenizer.apply_chat_template(
            request.original_messages, tokenize=True, return_tensors="pt"
        ).to(self.device)
        compressed_tokenized = self.reward_tokenizer.apply_chat_template(
            request.compressed_messages, tokenize=True, return_tensors="pt"
        ).to(self.device)
        original_score = self.reward_model(original_tokenized).logits[0][0].item()
        compressed_score = self.reward_model(compressed_tokenized).logits[0][0].item()
        logger.info(
            f"original_score: {original_score}, compressed_score: {compressed_score}"
        )
        compress_gain = (compressed_score - original_score) / (
            abs(original_score) + 1e-6
        ) + 1
        score = self.original_base_reward * min(1, compress_gain)
        logger.info(f"compress_gain: {compress_gain}, score: {score}")
        return ScoringResponse(score=score)


def start_server():
    app = App()
    uvicorn.run(
        app.app,
        host=os.getenv("CONDENSES_SCORING_HOST", "0.0.0.0"),
        port=int(os.getenv("CONDENSES_SCORING_PORT", 8000)),
    )


if __name__ == "__main__":
    start_server()
