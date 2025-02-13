from fastapi import FastAPI
import os
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import uvicorn
from loguru import logger
import numpy as np
import tiktoken
from .schemas import ScoringRequest, ScoringResponse
from transformers import pipeline

# Constants
TEMPERATURE = float(os.getenv("NCS_TEMPERATURE", 1.5))
COMPRESSION_SCALE = float(os.getenv("NCS_COMPRESSION_SCALE", 0.2))
TIKTOKEN_MODEL = os.getenv("NCS_TIKTOKEN_MODEL", "gpt-4o")
DEFAULT_HOST = os.getenv("NCS_SCORING_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("NCS_SCORING_PORT", 8000))


def sigmoid(x: float) -> float:
    """Apply sigmoid function with temperature scaling."""
    return 1 / (1 + np.exp(-x / TEMPERATURE))


class ScoringModel:
    def __init__(self):
        model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prompt_guard = pipeline(
            "text-classification",
            model="katanemo/Arch-Guard",
            device_map=self.device,
        )

    def guarding(self, prompt: str) -> bool:
        result = self.prompt_guard(prompt)
        logger.info(f"Prompt guard result: {result} | prompt: {prompt[:32]}")
        return result[0]["label"] == "JAILBREAK"

    @torch.no_grad()
    def score_messages(self, messages) -> float:
        """Score a set of messages using the reward model."""
        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.device)
        return self.model(tokenized).logits[0][0].item()


class App:
    def __init__(self):
        self.app = FastAPI()
        self.llm_client = OpenAI()
        self.scoring_model = ScoringModel()
        self.original_base_reward = 0.75
        self.tiktoken_encoder = tiktoken.encoding_for_model(TIKTOKEN_MODEL)

        self.app.add_api_route(
            "/api/scoring",
            self.api_scoring,
            methods=["POST"],
            response_model=ScoringResponse,
        )

    def calculate_compression_rate(
        self, compressed_msg: str, original_msg: str
    ) -> float:
        """Calculate the compression rate between two messages."""
        n_compressed = len(self.tiktoken_encoder.encode(compressed_msg))
        n_original = len(self.tiktoken_encoder.encode(original_msg))
        return n_compressed / n_original

    def api_scoring(self, request: ScoringRequest) -> ScoringResponse:
        # Find compressed message and calculate compression rate
        for comp_msg, orig_msg in zip(
            request.compressed_messages, request.original_messages
        ):
            if comp_msg.is_compressed:
                if self.scoring_model.guarding(comp_msg.content):
                    logger.warning(
                        f"Prompt guard detection | content_preview: {comp_msg.content[:100]} | event: prompt_guard_failed"
                    )
                    return ScoringResponse(score=0.0)
                compressed_rate = self.calculate_compression_rate(
                    comp_msg.content, orig_msg.content
                )
                logger.info(
                    f"Compression rate calculated | value: {compressed_rate} | event: compression_rate"
                )
                break

        # Calculate scores
        original_score = self.scoring_model.score_messages(request.original_messages)
        compressed_score = self.scoring_model.score_messages(
            request.compressed_messages
        )

        logger.info(
            f"Scoring completed | original_score: {original_score} | compressed_score: {compressed_score} | event: scoring_complete"
        )

        # Calculate final score
        compress_gain = sigmoid(compressed_score) / sigmoid(original_score)
        score = (
            self.original_base_reward * compress_gain * (1 - COMPRESSION_SCALE)
            + COMPRESSION_SCALE * compressed_rate
        )

        logger.info(
            f"Final score calculated | compress_gain: {compress_gain} | score: {score} | event: final_score"
        )
        return ScoringResponse(score=score)


def start_server():
    app = App()
    uvicorn.run(
        app.app,
        host=os.getenv("NCS_SCORING_HOST", DEFAULT_HOST),
        port=int(os.getenv("NCS_SCORING_PORT", DEFAULT_PORT)),
    )


if __name__ == "__main__":
    start_server()
