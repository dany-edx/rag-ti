import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

logger = logging.getLogger(__name__)
def _substack_reader(soup: Any, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {
        "Title of this Substack post": soup.select_one("h1.post-title").getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, extra_info


def _readthedocs_reader(soup: Any, url: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadTheDocs documentation site"""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a", {"class": "reference internal"})
    rtd_links = []

    for link in links:
        rtd_links.append(link["href"])
    for i in range(len(rtd_links)):
        if not rtd_links[i].startswith("http"):
            rtd_links[i] = urljoin(url, rtd_links[i])

    texts = []
    for doc_link in rtd_links:
        page_link = requests.get(doc_link)
        soup = BeautifulSoup(page_link.text, "html.parser")
        try:
            text = soup.find(attrs={"role": "main"}).get_text()

        except IndexError:
            text = None
        if text:
            texts.append("\n".join([t for t in text.split("\n") if t]))
    return "\n".join(texts), {}


def _readmedocs_reader(
    soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadMe documentation site"""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a")
    docs_links = [link["href"] for link in links if "/docs/" in link["href"]]
    docs_links = list(set(docs_links))
    for i in range(len(docs_links)):
        if not docs_links[i].startswith("http"):
            docs_links[i] = urljoin(url, docs_links[i])

    texts = []
    for doc_link in docs_links:
        page_link = requests.get(doc_link, headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        soup = BeautifulSoup(page_link.text, "html.parser")
        try:
            text = ""
            for element in soup.find_all("article", {"id": "content"}):
                for child in element.descendants:
                    if child.name == "a" and child.has_attr("href"):
                        if include_url_in_text:
                            url = child.get("href")
                            if url is not None and "edit" in url:
                                text += child.text
                            else:
                                text += (
                                    f"{child.text} (Reference url: {doc_link}{url}) "
                                )
                    elif child.string and child.string.strip():
                        text += child.string.strip() + " "

        except IndexError:
            text = None
            logger.error(f"Could not extract text from {doc_link}")
            continue
        texts.append("\n".join([t for t in text.split("\n") if t]))
    return "\n".join(texts), {}


def _gitbook_reader(
    soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a ReadMe documentation site"""
    import requests
    from bs4 import BeautifulSoup

    links = soup.find_all("a")
    docs_links = [link["href"] for link in links if "/docs/" in link["href"]]
    docs_links = list(set(docs_links))
    for i in range(len(docs_links)):
        if not docs_links[i].startswith("http"):
            docs_links[i] = urljoin(url, docs_links[i])

    texts = []
    for doc_link in docs_links:
        page_link = requests.get(doc_link, headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        soup = BeautifulSoup(page_link.text, "html.parser")
        try:
            text = ""
            text = soup.find("main")
            clean_text = clean_text = ", ".join([tag.get_text() for tag in text])
        except IndexError:
            text = None
            logger.error(f"Could not extract text from {doc_link}")
            continue
        texts.append(clean_text)
    return "\n".join(texts), {}


DEFAULT_WEBSITE_EXTRACTOR: Dict[
    str, Callable[[Any, str], Tuple[str, Dict[str, Any]]]
] = {
    "substack.com": _substack_reader,
    "readthedocs.io": _readthedocs_reader,
    "readme.com": _readmedocs_reader,
    "gitbook.io": _gitbook_reader,
}


class BeautifulSoupWebReader(BaseReader):
    def __init__(
        self,
        website_extractor: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize with parameters."""
        self.website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR

    def load_data(
        self,
        urls: List[str],
        custom_hostname: Optional[str] = None,
        include_url_in_text: Optional[bool] = True,
    ) -> List[Document]:
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup

        documents = []
        for url in urls:
            try:
                page = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            except Exception:
                raise ValueError(f"One of the inputs is not a valid url: {url}")

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

            data = ""
            extra_info = {"URL": url}
            if hostname in self.website_extractor:
                data, metadata = self.website_extractor[hostname](
                    soup=soup, url=url, include_url_in_text=include_url_in_text
                )
                extra_info.update(metadata)

            else:
                data = soup.getText()

            documents.append(Document(text=data, extra_info=extra_info))

        return documents