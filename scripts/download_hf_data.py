# download_hf_data.py
# Script to download sample news articles from Hugging Face Datasets and save as .txt files for ingestion.

# Download and save 10 news articles from the 'ag_news' dataset
from datasets import load_dataset
import os

DATASET_NAME = "ag_news"
SPLIT = "train"
OUTPUT_DIR = "data"
NUM_ARTICLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading dataset {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"Saving {NUM_ARTICLES} articles to {OUTPUT_DIR}/ ...")
    count = 0
    for item in dataset:
        # ag_news has 'text' and 'description' fields
        text = item.get("text") or item.get("description") or ""
        if text.strip():
            with open(os.path.join(OUTPUT_DIR, f"news_article_{count}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
        if count >= NUM_ARTICLES:
            break
    print(f"Done. Saved {count} articles.")

if __name__ == "__main__":
    main()
