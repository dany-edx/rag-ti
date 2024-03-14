from typing import Any, Dict, List, Optional, Union
import requests
from llama_index.core import download_loader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

class RemoteDepthReader(BaseReader):
    def __init__(
        self,
        *args: Any,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        depth: int = 1,
        domain_lock: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.file_extractor = file_extractor
        self.depth = depth
        self.domain_lock = domain_lock

    def load_data(self, url: str) -> List[Document]:
        from tqdm.auto import tqdm

        """Parse whatever is at the URL.""" ""
        try:
            from llama_hub.utils import import_loader

            RemoteReader = import_loader("RemoteReader")
        except ImportError:
            RemoteReader = download_loader("RemoteReader")
        remote_reader = RemoteReader(file_extractor=self.file_extractor)
        documents = []
        links = self.get_links(url)
        urls = {-1: [url]}  # -1 is the starting point
        links_visited = []
        for i in range(self.depth + 1):
            urls[i] = []
            new_links = []
            print(f"Reading links at depth {i}...")
            for link in tqdm(links):
                """Checking if the link belongs the provided domain."""
                if (self.domain_lock and link.find(url) > -1) or (not self.domain_lock):
                    # print("Loading link: " + link)
                    links_visited.append(link)

        return links_visited

    @staticmethod
    def is_url(href) -> bool:
        """Check if a link is a URL."""
        return href.startswith("http")

    def get_links(self, url) -> List[str]:
        from urllib.parse import urljoin, urlparse, urlunparse

        from bs4 import BeautifulSoup

        """Get all links from a page."""
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        links = soup.find_all("a")
        result = []
        for link in links:
            if isinstance(link, str):
                href = link
            else:
                href = link.get("href")
            if href is not None:
                if not self.is_url(href):
                    href = urljoin(url, href)

            url_parsed = urlparse(href)
            url_without_query_string = urlunparse(
                (url_parsed.scheme, url_parsed.netloc, url_parsed.path, "", "", "")
            )

            if (
                url_without_query_string not in result
                and url_without_query_string
                and url_without_query_string.startswith("http")
            ):
                result.append(url_without_query_string)
        return result
        
# loader = RemoteDepthReader(depth = 0)
# documents = loader.load_data(url="https://docs.llamaindex.ai/en/latest/index.html#")