from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
from keys import OPEN_AI_API_KEY

nltk.download('punkt')  
client = OpenAI(api_key=OPEN_AI_API_KEY)


def get_openai_embedding(text: str) -> list[float]:
    """
    Fetches the embedding for a given text using OpenAI's embedding API.
    """
    if not text.strip(): 
        return []
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


def calculate_relevance(query: str, response: list[dict]) -> (float, list[float]):
    """
    Calculate the relevance score for each JSON chunk and the total relevance score.
    It calculates the relevance of each chunk with the query and then averages to find the overall similarity.
    """
    query_embedding = get_openai_embedding(query)
    chunk_relevances = []

    for chunk in response:
        content = " ".join(chunk.get(key, "") for key in chunk if isinstance(chunk[key], str))
        if content.strip():
            chunk_embedding = get_openai_embedding(content)
            relevance = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            chunk_relevances.append(relevance)

    total_relevance_score = sum(chunk_relevances) / len(chunk_relevances) if chunk_relevances else 0
    return total_relevance_score, chunk_relevances


def calculate_completeness(query: str, response: list[dict]) -> float:
    """
    Calculate how completely the response addresses the query.
    The function calculates how completely the response addresses the query 
    by comparing the tokens (words) in the query with the tokens in the response. 
    Completeness is quantified as the proportion of query tokens that are also present in the response.
    """
    query_tokens = set(word_tokenize(query.lower()))
    response_tokens = set()

    # Collect tokens from all substantial text content in the response
    for chunk in response:
        content = " ".join(chunk.get(key, "") for key in chunk if isinstance(chunk[key], str))
        response_tokens.update(word_tokenize(content.lower()))

    # Calculate token overlap
    completeness = len(query_tokens.intersection(response_tokens)) / len(query_tokens) if query_tokens else 0
    return completeness


def calculate_overlap(response: list[dict]) -> float:
    """
    Calculate the overlap score to penalize redundant information across chunks.
    The function calculates an overlap score to penalize redundant information across chunks 
    in a response. The score is inversely proportional to the amount of overlap, with more redundancy (high overlap) 
    resulting in a lower score.
    """
    chunk_embeddings = []

    # Collect embeddings for substantial text content
    for chunk in response:
        content = " ".join(chunk.get(key, "") for key in chunk if isinstance(chunk[key], str))
        if content.strip():
            chunk_embeddings.append(get_openai_embedding(content))

    # Calculate pairwise cosine similarity between all chunk embeddings
    overlap_penalty = 0
    num_pairs = 0

    for i in range(len(chunk_embeddings)):
        for j in range(i + 1, len(chunk_embeddings)):
            similarity = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[j]])[0][0]
            if similarity > 0.8:  # Threshold for overlap
                overlap_penalty += 1
            num_pairs += 1

    # Normalize overlap score (1 - overlap_penalty / max_possible_pairs)
    overlap_score = 1 - (overlap_penalty / num_pairs) if num_pairs > 0 else 1
    return overlap_score


def evaluate_response(query: str, response: list[dict]) -> dict:
    """
    Evaluate the response for relevance, completeness, and overlap.
    """
    relevance_score, chunk_relevances = calculate_relevance(query, response)
    completeness_score = calculate_completeness(query, response)
    overlap_score = calculate_overlap(response)
    weights = [0.5,0.3,0.2]
    final_score = weights[0]*relevance_score + weights[1]*completeness_score + weights[2]*overlap_score
    return {
        "total_relevance": relevance_score,
        "chunk_relevances": chunk_relevances,
        "completeness": completeness_score,
        "overlap_score": overlap_score,
        "final_score": final_score
    }


# Example usage
query = "How does quantum superposition differ from classical binary states?"
response = [
    {"header": "Quantum Superposition vs Classical Binary States"},
    {"paragraph": "Quantum superposition is a fundamental principle in quantum mechanics where a quantum system can exist in multiple states simultaneously, unlike classical systems which must be in one definite state at any given time. In quantum superposition, a system can be a linear combination of all its possible states, each with associated complex probability amplitudes. For example, a qubit (quantum bit) can be both 0 and 1 at the same time, whereas a classical bit can only be either 0 or 1."},
    {"metric": "2^N", "description": "The number of possible states a qubit can exist in, where N is the number of qubits"},
    {"paragraph": "The principle of quantum superposition arises from the linearity of the Schrödinger equation, which governs the behavior of quantum systems. This equation is a linear differential equation in time and position, meaning that any linear combination of its solutions is also a solution. Mathematically, this is expressed as the ability to form a new valid quantum state by combining different distinct states using complex coefficients."},
    {"table": [["Classical Bit", "Quantum Bit"], ["Can only be 0 or 1", "Can be both 0 and 1"], ["Deterministic", "Probabilistic"]]},
    {"web_query": "diagram of a qubit in superposition", "query_type": "image"}
]

metrics = evaluate_response(query, response)
print(metrics)
