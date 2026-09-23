from langchain_core.tools import tool


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in a text.
    """

    text = text.strip()

    if not text:
        return "Word count: 0"

    count = len(text.split())

    return f"Word count: {count}"


if __name__ == "__main__":

    result = word_counter.invoke(
        {
            "text": "LangChain agents can choose and use tools automatically."
        }
    )

    print(result)