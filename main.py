import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from calculator_tool import calculator
from titanic_tool import titanic_fare_predictor
from distance_tool import distance_converter
from word_tool import word_counter
from note_tool import note_manager


# =========================================================
# Configuration
# =========================================================

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

# True = show which tool the agent used
# False = show only final answer
SHOW_TOOL_TRACE = True


# =========================================================
# Check API key
# =========================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY was not found in the .env file."
    )


# =========================================================
# Gemini model
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME
)


# =========================================================
# Exactly 5 LangChain tools
# =========================================================

tools = [
    calculator,
    titanic_fare_predictor,
    distance_converter,
    word_counter,
    note_manager,
]


# =========================================================
# Create LangChain Agent
# =========================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant powered by LangChain.

You have exactly five tools:

1. calculator
   Use this for mathematical calculations.

2. titanic_fare_predictor
   Use this to predict Titanic passenger fare using
   the trained machine-learning model.

3. distance_converter
   Use this to convert between kilometers and miles.

4. word_counter
   Use this to count words in text.

5. note_manager
   Use this to save, read, or append text notes.

Rules:

- Use the appropriate tool whenever a request matches
  one of the available tools.
- Do not perform calculations manually when the
  calculator tool can do them.
- Do not estimate Titanic fares yourself. Always use
  the Titanic machine-learning tool.
- Complete all requested tasks when the user asks for
  multiple tasks.
- Keep answers clear, concise, and human-readable.
- Do not expose internal tool-call JSON unless needed.
- When a tool returns an error, explain the error
  clearly to the user.
""",
)


# =========================================================
# Convert AIMessage content into normal text
# =========================================================

def extract_text(message):
    """
    Convert LangChain/Gemini message content into
    human-readable plain text.
    """

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content)


# =========================================================
# Display tool execution in readable form
# =========================================================

def show_tool_trace(messages):
    """
    Display tool calls and results in a readable format.
    """

    used_tool = False

    for message in messages:

        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
        ):

            for tool_call in message.tool_calls:

                used_tool = True

                print(
                    f"\n[Tool] {tool_call['name']}"
                )

                print(
                    f"[Input] {tool_call['args']}"
                )

        if type(message).__name__ == "ToolMessage":

            print(
                f"[Result] {message.content}"
            )

    return used_tool


# =========================================================
# Show available tools
# =========================================================

def show_tools():

    print("\nAvailable Tools")
    print("-" * 45)

    print(
        "1. calculator"
        " - Mathematical calculations"
    )

    print(
        "2. titanic_fare_predictor"
        " - Titanic fare prediction"
    )

    print(
        "3. distance_converter"
        " - Kilometers / miles conversion"
    )

    print(
        "4. word_counter"
        " - Count words in text"
    )

    print(
        "5. note_manager"
        " - Save, read, and append notes"
    )


# =========================================================
# Main interactive agent
# =========================================================

def main():

    print("\n" + "=" * 55)
    print("LANGCHAIN AI AGENT")
    print("=" * 55)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Tools: {len(tools)}"
    )

    print(
        "\nCommands:"
        "\n  tools  - show available tools"
        "\n  clear  - clear conversation memory"
        "\n  exit   - close the agent"
    )

    print("=" * 55)

    conversation = []

    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()

            if not user_input:
                continue


            # ---------------------------------------------
            # Exit
            # ---------------------------------------------

            if user_input.lower() in [
                "exit",
                "quit",
                "bye",
            ]:

                print(
                    "\nAgent: Goodbye."
                )

                break


            # ---------------------------------------------
            # Show tools
            # ---------------------------------------------

            if user_input.lower() == "tools":

                show_tools()

                continue


            # ---------------------------------------------
            # Clear conversation
            # ---------------------------------------------

            if user_input.lower() == "clear":

                conversation = []

                print(
                    "\nAgent: Conversation memory cleared."
                )

                continue


            # ---------------------------------------------
            # Add user message
            # ---------------------------------------------

            conversation.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )


            previous_message_count = len(
                conversation
            )


            # ---------------------------------------------
            # Run LangChain agent
            # ---------------------------------------------

            result = agent.invoke(
                {
                    "messages": conversation
                }
            )


            updated_messages = result[
                "messages"
            ]


            # Only messages produced during this turn
            new_messages = updated_messages[
                previous_message_count:
            ]


            # ---------------------------------------------
            # Tool trace
            # ---------------------------------------------

            if SHOW_TOOL_TRACE:

                show_tool_trace(
                    new_messages
                )


            # ---------------------------------------------
            # Human-readable final answer
            # ---------------------------------------------

            final_message = updated_messages[-1]

            answer = extract_text(
                final_message
            )

            print(
                f"\nAgent: {answer}"
            )


            # ---------------------------------------------
            # Keep conversation memory
            # ---------------------------------------------

            conversation = updated_messages


        except KeyboardInterrupt:

            print(
                "\n\nAgent: Goodbye."
            )

            break


        except Exception as error:

            print(
                f"\nAgent Error: {error}"
            )


# =========================================================
# Start program
# =========================================================

if __name__ == "__main__":
    main()