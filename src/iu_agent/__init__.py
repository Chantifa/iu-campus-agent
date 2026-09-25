"""IU Campus Agent: a Claude-Code-style CLI agent with a RAG over IU course material."""

import os

# fastembed downloads public models from the Hugging Face hub. huggingface_hub reads these flags
# at import time, so they have to be set before any embedding library is imported: an expired
# token stored on the machine would otherwise turn public downloads into 401 errors, and the
# symlink warning on Windows is only noise.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "0.1.0"
