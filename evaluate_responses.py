import json
from typing import Dict, List, Any, Tuple
from api_utils import generate_and_get_document
from default_payload import DEFAULT_PAYLOAD


def extract_metrics_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract metric blocks from the generated document."""
    metrics = []
    for block in blocks:
        if "metric" in block:
            metrics.append({
                "metric": block["metric"],
                "description": block["description"],
                "citations": block.get("citations", [])
            })
    return metrics


def evaluate_query(query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate document for a query and extract its metrics."""
    # Generate document using existing API
    payload = {**DEFAULT_PAYLOAD, **{"user_query": query}}
    blocks = generate_and_get_document(payload)
    
    # Extract metrics
    metrics = extract_metrics_from_blocks(blocks)
    
    return blocks, metrics


def main():
    # Example usage
    query = "What are the key differences between classical and quantum computing?"
    blocks, metrics = evaluate_query(query)
    
    print(f"\nQuery: {query}")
    print("\nMetrics found:")
    for idx, metric in enumerate(metrics, 1):
        print(f"\n{idx}. Metric: {metric['metric']}")
        print(f"   Description: {metric['description']}")
        if metric['citations']:
            print(f"   Citations: {metric['citations']}")
    
    # Optionally save results
    results = {
        "query": query,
        "full_document": blocks,
        "metrics": metrics
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()