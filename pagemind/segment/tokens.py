CHARS_PER_TOKEN = 4

SECTION_CAP = 10_000   # tokens; hard upper bound per section
CHUNK_TARGET = 384     # tokens; nominal chunk size
CHUNK_OVERLAP = 50     # tokens; ~13% overlap between adjacent chunks

# Usable budget when the context window is nominally 32K (per [0009])
USABLE_BUDGET = 28_000


def count_tokens(text: str) -> int:
    """Backend-agnostic token estimate: ~4 chars per token."""
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def cap_chars() -> int:
    return SECTION_CAP * CHARS_PER_TOKEN


def chunk_chars() -> int:
    return CHUNK_TARGET * CHARS_PER_TOKEN


def overlap_chars() -> int:
    return CHUNK_OVERLAP * CHARS_PER_TOKEN
