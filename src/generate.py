import json
from pathlib import Path
import asyncio
from typing import List, Dict
import logging
import time
from langchain_openai import AzureChatOpenAI  # Use Azure OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from asyncio import Semaphore
import os
from system_prompt import SYSTEM_PROMPT

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint="https://biswa-m5bzgk11-eastus2.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview",
    deployment_name="gpt-4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.3
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 300  
MAX_TOKENS_PER_MINUTE = 150_000  
CONCURRENT_REQUESTS = 15  

# Load Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


async def generate_pairs_for_chunk(
    chunk: Dict, chat: AzureChatOpenAI, semaphore: Semaphore, token_counter: Dict[str, int]
) -> List[Dict]:
    """Generate instruction-answer pairs with rate limiting."""
    try:
        async with semaphore:
            estimated_tokens = len(chunk["content"]) // 4 + 500  

            current_minute = int(time.time() // 60)
            if current_minute not in token_counter:
                token_counter.clear()
                token_counter[current_minute] = 0

            if token_counter[current_minute] + estimated_tokens > MAX_TOKENS_PER_MINUTE:
                logger.info("Token limit approaching, waiting for next minute...")
                await asyncio.sleep(60 - time.time() % 60)
                token_counter.clear()

            token_counter[current_minute] = (
                token_counter.get(current_minute, 0) + estimated_tokens
            )

            logger.info(f"Processing chunk from: {chunk['metadata']['url']}")

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Generate instruction-answer pairs from this documentation:\n\n{chunk['content']}"
                ),
            ]

            chat = AzureChatOpenAI(
    azure_endpoint="https://biswa-m5bzgk11-eastus2.openai.azure.com",
    deployment_name="gpt-4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.3
)

            response = await chat.ainvoke(messages)

            pairs = []
            current_pair = {}
            in_code_block = False
            code_content = []

            for line in response.content.split("\n"):
                if line.strip().startswith("```"):
                    if in_code_block:
                        in_code_block = False
                        code_text = "\n".join(code_content)
                        if current_pair and "answer" in current_pair:
                            current_pair["answer"] += f"\n```{code_text}```\n"
                        code_content = []
                    else:
                        in_code_block = True
                        code_content = (
                            [line.strip()[3:]] if len(line.strip()) > 3 else []
                        )
                    continue

                if in_code_block:
                    code_content.append(line)
                    continue

                line = line.strip()
                if not line:
                    continue

                if line.startswith("Q: "):
                    if current_pair:
                        pairs.append(current_pair)
                    current_pair = {"instruction": line[3:], "answer": ""}
                elif line.startswith("A: ") and current_pair:
                    current_pair["answer"] = line[3:]
                elif current_pair and "answer" in current_pair:
                    current_pair["answer"] += f"\n{line}"

            if current_pair:
                pairs.append(current_pair)

            return pairs

    except Exception as e:
        logger.error(f"Error processing chunk {chunk['id']}: {str(e)}")
        return []


async def generate_dataset(
    input_file: Path, output_file: Path, chunk_limit: int = None
):
    """Generate dataset with parallel processing and rate limiting."""
    try:
        start_time = time.time()

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data["data"][:chunk_limit] if chunk_limit else data["data"]

        logger.info(f"Processing {len(chunks)} chunks with rate limiting:")
        logger.info(f"- Max requests per minute: {MAX_REQUESTS_PER_MINUTE}")
        logger.info(f"- Max tokens per minute: {MAX_TOKENS_PER_MINUTE}")
        logger.info(f"- Concurrent requests: {CONCURRENT_REQUESTS}")

        from langchain_openai import AzureChatOpenAI
        import os

        chat = AzureChatOpenAI(
    azure_endpoint="https://biswa-m5bzgk11-eastus2.openai.azure.com",
    deployment_name="gpt-4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.3
)


        semaphore = Semaphore(CONCURRENT_REQUESTS)
        token_counter = {}

        tasks = []
        for chunk in chunks:
            task = generate_pairs_for_chunk(chunk, chat, semaphore, token_counter)
            tasks.append(task)
            await asyncio.sleep(0.1)

        all_results = await asyncio.gather(*tasks)
        all_pairs = [pair for result in all_results for pair in result]

        output_data = {
            "train": all_pairs,
            "metadata": {
                "total_chunks_processed": len(chunks),
                "total_pairs_generated": len(all_pairs),
                "processing_time": time.time() - start_time,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
        logger.info(f"Generated {len(all_pairs)} pairs from {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise


if __name__ == "__main__":
    input_file = Path("data/firecrawl_chunked_dataset.json")
    output_file = Path("data/firecrawl_instructions.json")
    asyncio.run(generate_dataset(input_file, output_file))
