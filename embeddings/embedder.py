import os
import logging
import warnings

# Hide warnings
warnings.filterwarnings("ignore")

# Hide Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hide transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def generate_embeddings(texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )