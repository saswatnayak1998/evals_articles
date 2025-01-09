import logging
import ast
import json
import re
import traceback
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BLOCK_FIELDS = {
    "paragraph": {
        "paragraph": str,
        "citations": List[int],
    },
    "table": {
        "table": List[List[str]],
        "citations": List[int],
    },
    "web_query": {
        "web_query": str,
        "query_type": str,
    },
    "metric": {
        "metric": str,
        "description": str,
        "citations": List[int],
    },
    "header": {
        "header": str,
    },
    "ai_image": {
        "ai_image": str,
    },
    "tweet": {
        "tweet": str,
        "citations": List[int],
        "tweet_multimedia_type": str,
        "tweet_multimedia_description": str,
    },
}


def logging_wrapper(msg, log_level, extra_context):
    if log_level == "info":
        logger.info(msg, extra={"info_context": extra_context})
    elif log_level == "warning":
        logger.warning(msg, extra={"info_context": extra_context})
    elif log_level == "debug":
        logger.debug(msg, extra={"info_context": extra_context})


def get_key_type(text: str, key: str) -> List[str]:
    start = text.find(key) + len(key)
    initial_slice = text[start:]
    start = initial_slice.find(":")
    slice = initial_slice[start + 1 :].strip()
    if slice.startswith('"') or slice.startswith("'"):
        key_type = str
    elif slice.startswith("[["):
        key_type = List[list]
    elif slice.startswith("["):
        key_type = list
    elif slice.startswith("{"):
        key_type = dict
    elif slice.startswith("[{"):
        key_type = List[dict]
    elif any(slice.startswith(str(n)) for n in range(10)):
        key_type = int
    elif slice.startswith("False") or slice.startswith("True"):
        key_type = bool
    else:
        raise ValueError(f"Could not find a valid key type in the text: {text}")
    return key_type


def get_keys(text: str) -> List[str]:
    pattern = r"""(['"])([^'":]*)\1:"""
    matches = re.findall(pattern, text)
    # Note: list(set(matches)) can change the order of the keys
    keys = []
    for match in matches:
        key = match[1]
        if key not in keys:
            keys.append(key)
    return keys


def remove_numbers_in_parentheses(text: str):
    pattern = r"\(\d+\)"
    text_without_numbers = re.sub(pattern, "", text)
    while " ." in text_without_numbers:
        text_without_numbers = text_without_numbers.replace(" .", ".")
    while "  " in text_without_numbers:
        text_without_numbers = text_without_numbers.replace("  ", " ")
    return text_without_numbers


def reshape_list_elements(elements, n_columns):
    result = []
    while elements:
        result.append(elements[:n_columns])
        elements = elements[n_columns:]
    return result


def load(llm_resp):
    try:
        result = json.loads(llm_resp, strict=False)
    except Exception as e:
        result = None
    return result


def dict_fallback_1(llm_resp):
    start, end = llm_resp.find("{"), llm_resp.rfind("}")
    return load(llm_resp[start : end + 1])


def dict_fallback_2(llm_resp):
    try:
        result = ast.literal_eval(llm_resp)
    except Exception as e:
        result = None
    return result


def load_dict(llm_resp):
    for method in (load, dict_fallback_1, dict_fallback_2):
        result = method(llm_resp)
        if result:
            break

    return result


def list_fallback_1(llm_resp):
    new_resp = llm_resp
    if not new_resp.startswith("["):
        new_resp = "[" + new_resp
    if not new_resp.endswith("]"):
        new_resp += "]"
    try:
        result = json.loads(new_resp, strict=False)
    except Exception as e:
        result = None
    return result


def list_fallback_2(llm_resp):
    result = []
    try:
        while "{" in llm_resp and "}" in llm_resp:
            start, end = result.find("{"), result.find("}")
            element = json.loads(result[start : end + 1], strict=False)
            result.append(element)
            result = result[end + 2 :]
    except Exception as e:
        result = None
    return result


def load_list(llm_resp):
    for method in (load, list_fallback_1, list_fallback_2):
        result = method(llm_resp)
        if result:
            break

    return result


def find_integer(string):
    pattern = r"\d+"
    match = re.search(pattern, string)
    if match:
        return int(match.group())
    else:
        return None


class JSONParser:
    """
    Parsing issue with our LLMs are usually not due to missing fields, but
    the presence of prohibited characters, or missing delimiters. As a result,
    we can almost always parse for the desired fields by slicing intro the string.
    Currently this is implemented for flat dictionaries.
    """

    @classmethod
    def _log(cls, msg, log_level="info", extra_context={}):
        if not isinstance(extra_context, dict):
            extra_context = {}
        extra_context.update({"script": "llm_parsing"})

        logging_wrapper(msg, log_level, extra_context)

    @classmethod
    def _remove_escapes(cls, some_str):
        while "\\" in some_str:
            some_str = some_str.replace("\\", "")
        return some_str

    @classmethod
    def _post_process_key_value(cls, key_value):
        if isinstance(key_value, str):
            key_value = key_value.strip()
            for chars in ('",', '\\",', '\\"', '"'):
                key_value = key_value.strip()
                while key_value.startswith(chars):
                    key_value = key_value[len(chars) :]
                    key_value = key_value.strip()
                while key_value.endswith(chars):
                    key_value = key_value[: -len(chars)]
                    key_value = key_value.strip()
            key_value = remove_numbers_in_parentheses(key_value)
        return key_value

    @classmethod
    def _split_json_str(cls, json_str):
        result = []
        while "{" in json_str and "}" in json_str:
            start, end = json_str.find("{"), json_str.find("}")
            element = json_str[start : end + 1]
            result.append(element)
            json_str = json_str[end + 2 :]
        return result

    @classmethod
    def _evaluate_list(cls, value):
        # value = value.replace("\\n", "").replace("\n", "").replace("],]", "]]")
        try:
            result = ast.literal_eval(value)
        except Exception as e:
            if value.count('"') % 2 != 0:
                if "]" not in value:
                    value += '"]'
                else:
                    value = value.replace("]", '"]')
            elif "]" not in value:
                value += "]"
            try:
                result = ast.literal_eval(value)
            except Exception as e:
                cls._log(
                    f"Could not recover list from {value} due to {traceback.format_exc()}",
                    log_level="debug",
                )
                result = None
        return result

    @classmethod
    def _evaluate_dict(cls, value):
        str_repr = value
        try:
            result = json.loads(str_repr, strict=False)
        except Exception as e:
            keys = get_keys(str_repr)
            field_types = {k: get_key_type(str_repr, k) for k in keys}
            result = cls.parse_json_dict(str_repr, field_types=field_types)
        return result

    @classmethod
    def _evaluate_2d_array(cls, value):
        start, end = value.find("["), value.rfind("]")
        if end > start >= 0:
            value = value[start : end + 1]
        value = value.replace("\\n", "").replace("\n", "")
        value = value.replace("], ]", "]]").replace("],]", "]]")
        value = value.replace("]]]", "]]").replace("[[[", "[[")
        value = value.strip()
        if value.endswith(","):
            value = value[:-1]
        try:
            cls._log(f"Trying to evaluate table from: {value}")
            result = ast.literal_eval(value)
        except Exception as e:
            cls._log(
                f"Could not evaluate table on first pass due to: {traceback.format_exc()}",
                log_level="debug",
            )
            if not value.endswith("]]"):
                if value.endswith("]"):
                    value += "]"
                elif value.endswith('"'):
                    value += "]]"
                else:
                    value += '"]]'
            elif not value.endswith('"]]'):
                value = value.replace("]]", '"]]')

            if value.count("]") == value.count("[") - 1:
                value += "]"

            try:
                cls._log(f"Retryig to evaluate table from: {value}")
                result = ast.literal_eval(value)
            except Exception as e:
                cls._log(
                    f"Could not recover table from {value} due to {traceback.format_exc()}",
                    log_level="debug",
                )
                result = None
        cls._log(f"Table parsed to type: {type(result).__name__}")
        if isinstance(result, list) and len(result) == 1:
            elements = result[0]
            n_elements = len(elements)
            if n_elements % 2 == 0 and n_elements % 3 != 0:
                result = reshape_list_elements(elements, 2)
            elif n_elements % 3 == 0:
                result = reshape_list_elements(elements, 3)
        return result

    @classmethod
    def _evaluate_list_of_dict(cls, value: str):
        # note currently assumes key values are strings
        try:
            result = cls._evaluate_list(value)
            if not result:
                raise Exception("Primary method failed.")
        except Exception as e:
            cls._log(
                f"Could not evaluate list of dict using primary method due to: {traceback.format_exc()}",
                log_level="debug",
            )
            try:
                keys = get_keys(value)
                field_types = {key: str for key in keys}
                result = []
                while value and all([f'"{key}"' in value for key in keys]):
                    field = f'"{keys[-1]}"'
                    result.append(JSONParser.parse_json_dict(value, field_types))
                    start_of_next_slice = value.find(field)
                    if start_of_next_slice == -1:
                        value = ""
                    else:
                        value = value[start_of_next_slice + len(field) :]

            except Exception as e:
                cls._log(
                    f"Could not evaluate list of dict using fallback method due to: {traceback.format_exc()}",
                    log_level="debug",
                )
                result = None

        return result

    @classmethod
    def _extract_slice(cls, json_str, key, keys_list, value_type):
        if key not in json_str:
            if key not in (
                "source_justification",
                "source_ID",
                "source_IDs",
            ):
                cls._log(
                    f"Key {key} was not in json_str: {json_str}", log_level="warning"
                )
            return None

        # generic slicing
        key_index = keys_list.index(key)

        if f'"{key}"' in json_str:
            quote_delim = '"'
        elif f"'{key}'" in json_str:
            quote_delim = "'"
        else:
            raise ValueError(f"""Could not find '{key}' or "{key}" in {json_str}.""")

        slice_start = json_str.find(f"{quote_delim}{key}{quote_delim}")
        if key_index == len(keys_list) - 1:
            slice_end = len(json_str)
        else:
            i = 1
            slice_end = json_str.find(
                f"{quote_delim}{keys_list[key_index + i]}{quote_delim}"
            )
            while slice_end == -1 and key_index + i < len(keys_list) - 1:
                i += 1
                slice_end = json_str.find(
                    f"{quote_delim}{keys_list[key_index + i]}{quote_delim}"
                )
            if slice_end == -1:
                slice_end = len(json_str)
        if slice_start == -1:
            cls._log(f"Could not find {key} in {json_str}", log_level="debug")
            slice_start = 0
        sliced_json_str = json_str[slice_start:slice_end]

        # cls._log(
        #     f"Generic slice for {key} ({slice_start}, {slice_end}): {sliced_json_str}"
        # )

        # type specific slicing
        if value_type == str:
            sliced_json_str = sliced_json_str[sliced_json_str.find(":") :].strip()
            if "'" not in sliced_json_str:
                char = '"'
            elif '"' not in sliced_json_str:
                char = "'"
            elif sliced_json_str.find('"') < sliced_json_str.find("'"):
                char = '"'
            else:
                char = "'"
            start = sliced_json_str.find(char) + 1
            end = sliced_json_str.rfind(char)
        elif value_type in (int, bool):
            start = sliced_json_str.find(":") + 1
            if key_index < len(keys_list) - 1 and any(
                [key in sliced_json_str for key in keys_list[key_index + 1 :]]
            ):
                end = sliced_json_str.rfind(",")
            else:
                end = sliced_json_str.rfind("}")

        elif value_type in (list, List[dict], List[int], List[str]):
            if "[" not in sliced_json_str or "]" not in sliced_json_str:
                cls._log(
                    f"List delimiters for {key} not in slice: {sliced_json_str}",
                    log_level="debug",
                )
            sliced_json_str = sliced_json_str.replace("{", "").replace("}", "")
            start = sliced_json_str.find("[")
            end = sliced_json_str.find("]") + 1
        elif value_type == Dict[str, Any]:
            start = sliced_json_str.find("{")
            first_right_bracket = sliced_json_str.find("}") + 1
            end = second_right_bracket = (
                sliced_json_str.find("}", first_right_bracket) + 1
            )
        elif value_type in (dict, Dict):
            if "{" not in sliced_json_str or "}" not in sliced_json_str:
                cls._log(
                    f"Dict delimiters for {key} not in slice: {sliced_json_str}",
                    log_level="debug",
                )
            start = sliced_json_str.find("{")
            end = sliced_json_str.find("}") + 1
        elif value_type in (List[list], List[List[str]]):
            sliced_json_str = sliced_json_str.replace("{", "").replace("}", "")
            start = sliced_json_str.find("[[")
            if start == -1:
                start = sliced_json_str.find("[ [")
            if start == -1:
                start = sliced_json_str.find("[")
            end = sliced_json_str.find("]]") + 2
            if end == 1:
                end = sliced_json_str.find("],]") + 3
            if end == 2:
                end = sliced_json_str.find("], ]") + 4
            if end == 3:
                end = len(sliced_json_str)
        else:
            raise ValueError(
                f"The type {value_type} is not supported by the JSON parser."
            )

        if start == -1:
            cls._log(f"Could not find start for {key} in {json_str}", log_level="debug")
            start = 0
        if end <= start:
            end = len(sliced_json_str)

        slice = sliced_json_str[start:end].strip()

        return slice

    @classmethod
    def _recover_key_value(cls, json_str, key, keys_list, value_type):
        slice_containing_value = cls._extract_slice(
            json_str, key, keys_list, value_type
        )
        cls._log(f"Slice for {key}: {slice_containing_value}")
        if not slice_containing_value:
            cls._log(
                f"Could not recover {key} from {json_str} because slice {slice_containing_value} is missing the key field",
                log_level="warning",
            )
            key_value = None
        elif value_type == str:
            key_value = slice_containing_value
        elif value_type == int:
            key_value = find_integer(slice_containing_value)
        elif value_type == bool:
            key_value = True if "true" in slice_containing_value.lower() else False
        elif value_type in (list, List[str], List[int]):
            key_value = cls._evaluate_list(slice_containing_value)
        elif value_type in (dict, Dict[str, Any]):
            key_value = cls._evaluate_dict(slice_containing_value)
        elif value_type in (List[list], List[List[int]], List[List[str]]):
            key_value = cls._evaluate_2d_array(slice_containing_value)
        elif value_type == List[dict]:
            key_value = cls._evaluate_list_of_dict(slice_containing_value)
        else:
            cls._log(f"Encountered an invalid value for {key}: {value_type}")
            key_value = None
        return key_value

    @classmethod
    def _recover_json(cls, json_str, field_types):
        keys_list = list(field_types.keys())
        recovered_json = {
            key: cls._recover_key_value(json_str, key, keys_list, value_type)
            for key, value_type in field_types.items()
        }
        return recovered_json

    @classmethod
    def parse_json_dict(cls, json_str, field_types) -> Dict[str, any]:
        json_str = json_str.replace("\\_", "_")
        result = cls._recover_json(json_str, field_types)
        if result:
            result = {
                cls._remove_escapes(k): cls._post_process_key_value(v)
                for k, v in result.items()
            }
        cls._log(f"Parsed {json_str} as {result}")
        return result

    @classmethod
    def parse_json_list(cls, json_str, field_types) -> List[Dict[str, any]]:
        json_str = json_str.replace("\\_", "_")

        result = None

        try:
            result = load_list(json_str)
            if not result or not isinstance(result, list):
                raise Exception
        except Exception as e:
            try:
                json_substrs = cls._split_json_str(json_str)
                result = [
                    cls._recover_json(json_substr, field_types)
                    for json_substr in json_substrs
                ]
                cls._log(f"recovered_result: {result}")
            except Exception as e:
                cls._log(
                    f"Failed to get result from {json_str} due to: {traceback.format_exc()}",
                    log_level="debug",
                )
                result = None

        if not isinstance(result, list):
            cls._log(
                f"parsed_json_list but retreived a: {type(result).__name__}",
                log_level="debug",
            )
            result = [result]

        if result:
            result = [
                {
                    cls._remove_escapes(k): cls._post_process_key_value(v)
                    for k, v in element.items()
                }
                for element in result
            ]

        cls._log(f"Parsed {json_str} as {result}")

        return result
