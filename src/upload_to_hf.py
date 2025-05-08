from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi, create_repo
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_dataset(input_file: Path) -> DatasetDict:
    """Prepare the dataset for HuggingFace Hub upload."""
    try:
        logger.info(f"Loading dataset from {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to HF Dataset format
        train_data = data["train"]

        # Create Dataset object
        dataset = Dataset.from_dict(
            {
                "instruction": [item["instruction"] for item in train_data],
                "answer": [item["answer"] for item in train_data],
            }
        )

        # Split into train/validation sets (optional)
        splits = dataset.train_test_split(
            test_size=0.1, seed=42, shuffle=True  # 10% for validation
        )

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {"train": splits["train"], "validation": splits["test"]}
        )

        logger.info(f"Dataset prepared:")
        logger.info(f"- Total examples: {len(dataset)}")
        logger.info(f"- Training examples: {len(splits['train'])}")
        logger.info(f"- Validation examples: {len(splits['test'])}")

        return dataset_dict

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise


def push_to_hub(
    dataset_dict: DatasetDict, repo_name: str, token: str = None, private: bool = False
):
    """Push the dataset to HuggingFace Hub with versioning."""
    try:
        # Login to Hugging Face
        api = HfApi(token=token)
        if token:
            login(token)
            logger.info("Logged in to Hugging Face Hub")

        # Ensure repository exists
        try:
            api.repo_info(repo_id=repo_name, repo_type="dataset")
            logger.info(f"Repository {repo_name} found")
        except Exception:
            logger.info(f"Creating new repository: {repo_name}")
            create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=private,
                token=token,
                exist_ok=True,
            )

        # Create repository name
        repo_id = f"{repo_name}"

        logger.info(f"Pushing dataset to {repo_id}")
        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            commit_message="Upload Firecrawl instruction dataset",
        )

        logger.info(f"Dataset successfully pushed to {repo_id}")
        logger.info(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        logger.error(f"Error pushing to hub: {str(e)}")
        raise


def main():
    """Main function to prepare and upload dataset."""
    # Configuration
    input_file = Path("data/firecrawl_instructions.json")
    repo_name = "swadhin42/unsloth-instructions"  # Replace with your username

    # Get HF token from environment or input
    import os

    token = "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"
    if not token:
        token = input("Enter your HuggingFace token: ")

    # Prepare and push dataset
    dataset_dict = prepare_dataset(input_file)
    push_to_hub(dataset_dict, repo_name, token, private=False)


if __name__ == "__main__":
    main()
