"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import os
from typing import List, Optional

import openai.types.chat

from adtools.lm.lm_base import LanguageModel


class OpenAIAPI(LanguageModel):
    def __init__(
        self,
        model: str,
        base_url: str = None,
        api_key: str = None,
        **openai_init_kwargs,
    ):
        super().__init__()
        # If base_url is set to None, find 'OPENAI_BASE_URL' in environment variables
        if base_url is None:
            if "OPENAI_BASE_URL" not in os.environ:
                raise RuntimeError(
                    'If "base_url" is None, the environment variable OPENAI_BASE_URL must be set.'
                )
            else:
                base_url = os.environ["OPENAI_BASE_URL"]

        # If api_key is set to None, find 'OPENAI_API_KEY' in environment variables
        if api_key is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError('If "api_key" is None, OPENAI_API_KEY must be set.')
            else:
                api_key = os.environ["OPENAI_API_KEY"]

        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, **openai_init_kwargs
        )

    def chat_completion(
        self,
        message: str | List[openai.types.chat.ChatCompletionMessageParam],
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """Send a chat completion query with OpenAI format to the vLLM server.
        Return the response content.

        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
        """
        if isinstance(message, str):
            message = [{"role": "user", "content": message.strip()}]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=message,
            stream=False,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            *args,
            **kwargs,
        )
        return response.choices[0].message.content

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """Generate embeddings for the given text(s) using the model specified during initialization.

        Args:
            text: The text or a list of texts to embed.
            dimensions: The number of dimensions for the output embeddings.
            timeout_seconds: The timeout seconds.

        Returns:
            The embedding for the text, or a list of embeddings for the list of texts.
        """
        is_str_input = isinstance(text, str)
        if is_str_input:
            text = [text]

        # Prepare arguments for the OpenAI API call
        api_kwargs = {
            "input": text,
            "model": self._model,
        }
        if dimensions is not None:
            api_kwargs["dimensions"] = dimensions
        if timeout_seconds is not None:
            api_kwargs["timeout"] = timeout_seconds

        api_kwargs.update(kwargs)

        response = self._client.embeddings.create(**api_kwargs)

        embeddings = [item.embedding for item in response.data]

        if is_str_input:
            return embeddings[0]
        return embeddings
