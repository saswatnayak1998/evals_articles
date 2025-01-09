import re
import requests
import copy
import sys
import time
import json
from typing import Dict, List, Any

from keys import CAPITOL_API_KEY
from json_parser import JSONParser, BLOCK_FIELDS
from default_payload import DEFAULT_PAYLOAD

CAPITOL_URL = "https://hackathon.capitol.ai"


def get_block_fields(block_str: str) -> dict[str, Any]:
    for block, block_fields in BLOCK_FIELDS.items():
        if f'"{block}"' in block_str:
            return block_fields
    raise ValueError(f"Could not parse: {block_str}")


def generate_document(user_config_params: dict[str, Any]) -> dict[str, Any]:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    llm_endpoint_payload = {
        "params": {"external_id": ""},
        "user_config_params": user_config_params,
    }
    response = requests.post(
        f"{CAPITOL_URL}/llm", headers=headers, json=llm_endpoint_payload
    )
    response_json = response.json()
    return response_json


def is_document_gen_complete(external_id: str) -> bool:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    response = requests.get(f"{CAPITOL_URL}/attributes/{external_id}", headers=headers)
    if response.status_code != 200:
        print("Failed to poll document generation")
        return False
    response_json = response.json()
    return not response_json["is_generating"]


def wait_for_document_completion(external_id: str, timeout_minutes: int = 5) -> bool:
    duration_seconds = timeout_minutes * 60
    end_time = time.monotonic() + duration_seconds
    print(f"polling document generation for external id: {external_id}")
    while time.monotonic() < end_time:
        document_gen_completed = is_document_gen_complete(external_id)
        print(f"document generation complete: {document_gen_completed}")
        if document_gen_completed:
            break
        time.sleep(15)
    return document_gen_completed


def get_raw_llm_content_response(external_id: str, draft_id: str) -> Dict[str, Any]:
    headers = {"CAP-LLM-API-KEY": CAPITOL_API_KEY}
    response = requests.post(
        f"{CAPITOL_URL}/fetch_raw_llm_response",
        headers=headers,
        json={"external_id": external_id, "draft_id": draft_id},
    )

    if response.status_code != 200:
        print(
            f"error in retrieving raw llm response for external id: {external_id} and draft id: {draft_id}\nresponse status code: {response.status_code}"
        )
        sys.exit(0)
    response_json = response.json()
    # print(json.dumps(response_json, indent=4))
    return response_json


def parse_raw_llm_response(raw_llm_response_api_data: Dict[str, Any]) -> List[str]:
    model_used = raw_llm_response_api_data["model_used"]
    raw_response_string = raw_llm_response_api_data["data"]
    pattern = r"\{.*?\}"
    raw_llm_blocks = re.findall(pattern, raw_response_string, re.DOTALL)
    return model_used, raw_llm_blocks


def generate_and_get_document(
    user_config_params: dict[str, Any], timeout_minutes: int = 10
) -> List[Dict[str, Any]]:
    response_json = generate_document(user_config_params)
    print(json.dumps(response_json, indent=4))
    external_id = response_json["external_id"]
    draft_id = response_json["draft_id"]
    document_gen_completed = wait_for_document_completion(external_id, timeout_minutes)
    if not document_gen_completed:
        print(f"Document generation did not complete in {timeout_minutes} minutes.")
        sys.exit(0)
    raw_response = get_raw_llm_content_response(external_id, draft_id)
    model_used, raw_llm_json_blocks = parse_raw_llm_response(raw_response)
    parsed_blocks = [
        JSONParser.parse_json_dict(block_str, field_types=get_block_fields(block_str))
        for block_str in raw_llm_json_blocks
    ]
    return parsed_blocks


def test_api():
    payload = {**DEFAULT_PAYLOAD, **{"user_query": "explain string theory"}}
    parsed_blocks = generate_and_get_document(payload)
    print(json.dumps(parsed_blocks, indent=4))
