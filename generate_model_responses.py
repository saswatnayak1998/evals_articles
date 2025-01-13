import numpy as np
import json
import concurrent.futures
import copy
from typing import Any, List, Dict
import logging

from api_utils import generate_and_get_document
from default_payload import DEFAULT_PAYLOAD
from evaluation_utils import evaluate_response  

MAX_WORKERS = 8

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_payload(user_query: str) -> Dict[str, Any]:
    """
    Build the payload for the API request.
    """
    payload = copy.deepcopy(DEFAULT_PAYLOAD)
    payload["user_query"] = user_query
    return payload


def sample_queries(seed: int = 0, n_samples_per_subject: int = 1) -> List[str]:
    """
    Sample queries from the JSON file.

    Args:
        seed (int): Seed for reproducibility.
        n_samples_per_subject (int): Number of queries to sample per subject.

    Returns:
        List[str]: A list of sampled queries.
    """
    np.random.seed(seed)
    sampled_queries = []

    try:
        with open("./example_queries.json", "r") as file:
            queries_by_subject = json.load(file)
    except FileNotFoundError:
        logging.error("example_queries.json file not found.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON in example_queries.json.")
        raise

    for subject, subject_queries in queries_by_subject.items():
        sampled_query = np.random.choice(
            list(subject_queries), size=n_samples_per_subject, replace=False
        )[0]
        sampled_queries.append(sampled_query)
        logging.info(f'Sampled "{sampled_query}" for subject "{subject}"')

    return sampled_queries


def generate_responses(queries: List[str]) -> None:
    """
    Generate responses for the given queries and calculate evaluation metrics.

    Args:
        queries (List[str]): List of queries to process.
    """
    models_responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {
            executor.submit(generate_and_get_document, build_payload(query)): query
            for query in queries
        }

        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                document = future.result()
                metrics = evaluate_response(query, document)

                models_responses.append({
                    "query": query,
                    "document": document,
                    "metrics": metrics
                })

                logging.info(f"Processed query: {query}")

            except Exception as e:
                logging.error(f"Error processing query '{query}': {e}")

    try:
        with open("./model_responses.json", "w") as f:
            json.dump(models_responses, f, indent=4)
            logging.info("Saved model responses to model_responses.json.")
    except Exception as e:
        logging.error(f"Error saving model responses: {e}")
        raise


if __name__ == "__main__":
    try:
        queries = sample_queries()
        generate_responses(queries)
    except Exception as e:
        logging.critical(f"Critical error: {e}")
