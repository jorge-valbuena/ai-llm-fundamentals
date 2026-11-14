"""LLM Autocomplete Demo — sends an unfinished sentence to Azure OpenAI and prints the completion."""

import os
from openai import OpenAI
from dotenv import load_dotenv


def get_inputs() -> str:
    """Return the unfinished input sentence for autocomplete prompting."""
    return "Who is Jorge Valbuena  "


def create_client(
    endpoint_url: str,
    api_key: str,
) -> OpenAI:
    """Build an OpenAI client pointed at an Azure AI Foundry v1-compatible endpoint."""
    return OpenAI(
        base_url=endpoint_url,
        api_key=api_key,
    )


def get_completion(input_text: str, client: OpenAI, deployment_name: str) -> str:
    """Send *input_text* to the chat model and return only its reply text."""
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": input_text}],
        temperature=0,
    )
    return response.choices[0].message.content


def print_result(input_text: str, response_text: str) -> None:
    """Print a clean, formatted demo output to stdout."""
    print("--- Completion Demo ---")
    print(f"Input: \"{input_text}\"")
    print("LLM Response:", response_text)


def main() -> None:
    load_dotenv()

    endpoint_url = os.environ["ENDPOINT_URL"]
    deployment_name = os.environ["DEPLOYMENT_NAME"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    client = create_client(endpoint_url=endpoint_url, api_key=api_key)

    input_text = get_inputs()
    response_text = get_completion(input_text=input_text, client=client, deployment_name=deployment_name)

    print_result(input_text=input_text, response_text=response_text)


if __name__ == "__main__":
    main()
