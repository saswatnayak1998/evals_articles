import openai
from keys import OPEN_AI_API_KEY


def check_openai_api_key():
    client = openai.OpenAI(api_key=OPEN_AI_API_KEY)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


if __name__ == "__main__":
    print(check_openai_api_key())
