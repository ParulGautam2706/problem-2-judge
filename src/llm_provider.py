import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMProvider:
    """
    OpenAI-compatible LLM provider.

    Supported providers:
        groq
        openai

    Example for Groq:

        JUDGE_PROVIDER=groq
        JUDGE_MODEL=llama-3.1-8b-instant
        GROQ_API_KEY=...

    Example for OpenAI:

        JUDGE_PROVIDER=openai
        JUDGE_MODEL=gpt-4o-mini
        OPENAI_API_KEY=...
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (
            provider
            or os.getenv("JUDGE_PROVIDER", "groq")
        ).lower()

        self.model = (
            model
            or os.getenv("JUDGE_MODEL")
            or self._default_model()
        )

        self.client = self._create_client()

    def _default_model(self):
        if self.provider == "groq":
            return "llama-3.1-8b-instant"

        if self.provider == "openai":
            return "gpt-4o-mini"

        raise ValueError(
            f"Unsupported provider: {self.provider}"
        )

    def _create_client(self):

        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY environment variable is not set."
                )

            return OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is not set."
                )

            return OpenAI(api_key=api_key)

        raise ValueError(
            f"Unsupported provider: {self.provider}"
        )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 350,
        retries: int = 3,
    ) -> LLMResponse:

        last_error = None

        for attempt in range(retries):

            start = time.perf_counter()

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )

                latency = time.perf_counter() - start

                text = ""

                if response.choices:
                    text = (
                        response.choices[0]
                        .message
                        .content
                        or ""
                    )

                usage = getattr(response, "usage", None)

                input_tokens = 0
                output_tokens = 0

                if usage:
                    input_tokens = (
                        getattr(
                            usage,
                            "prompt_tokens",
                            0,
                        )
                        or 0
                    )

                    output_tokens = (
                        getattr(
                            usage,
                            "completion_tokens",
                            0,
                        )
                        or 0
                    )

                return LLMResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=round(latency, 4),
                )

            except Exception as exc:

                last_error = exc

                error_text = str(exc).lower()

                is_rate_limit = (
                    "rate limit" in error_text
                    or "rate_limit" in error_text
                    or "429" in error_text
                    or "tokens per day" in error_text
                    or "tokens per minute" in error_text
                )

                if not is_rate_limit:
                    raise

                # Daily quota cannot be fixed by retrying.
                if "tokens per day" in error_text:

                    raise RuntimeError(
                        "\n"
                        "========================================\n"
                        "LLM DAILY TOKEN LIMIT REACHED\n"
                        "========================================\n"
                        f"Provider : {self.provider}\n"
                        f"Model    : {self.model}\n\n"
                        "Do not retry repeatedly.\n"
                        "Either wait for the quota reset or change\n"
                        "the judge model/provider.\n\n"
                        "Recommended:\n"
                        "  JUDGE_PROVIDER=groq\n"
                        "  JUDGE_MODEL=llama-3.1-8b-instant\n"
                        "\n"
                        f"Original error:\n{exc}\n"
                    ) from exc

                # Temporary 429.
                wait_seconds = 2 ** attempt

                print(
                    f"[Rate limit] "
                    f"waiting {wait_seconds}s "
                    f"before retry "
                    f"{attempt + 1}/{retries}..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            f"LLM request failed after {retries} retries: "
            f"{last_error}"
        )