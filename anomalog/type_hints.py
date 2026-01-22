from typing import Annotated
from urllib.parse import urlparse

from beartype.vale import Is

IsLen32 = Is[lambda s: len(s) == 32]


def _is_hex(s: str) -> bool:
    try:
        bytes.fromhex(s)
        return True
    except ValueError:
        return False


IsHex = Is[_is_hex]

MD5Hex = Annotated[str, IsLen32 & IsHex]


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


URL = Annotated[str, Is[_is_valid_url]]
