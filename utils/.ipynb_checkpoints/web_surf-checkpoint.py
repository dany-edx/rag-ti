"""Google Search tool spec."""

import urllib.parse
from typing import Optional
import json
import requests
from llama_index.core import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}"
)


class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""

    spec_functions = ["google_search"]

    def __init__(self, key: str, engine: str, num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def google_search(self, query: str):
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        response = requests.get(url)
        return [Document(text=response.text)]