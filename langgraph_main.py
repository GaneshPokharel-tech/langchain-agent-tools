import os

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

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

# True  -> show tool execution
# False -> show only final answer
SHOW_TOOL_TRACE = True


# =========================================================
# API Key Check
# =========================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY was not found in .env"
    )


# =========================================================
# System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are a precise LangGraph tool-using AI assistant.

You have exactly five tools:

1. calculator
   Use for mathematical calculations.

2. titanic_fare_predictor
   Use only to predict Titanic passenger fares.

   Required inputs:
   - pclass: 1, 2, or 3
   - sex: male or female
   - age
   - embarked: S, C, or Q
   - family_size

3. distance_converter
   Use to convert between kilometers and miles.

4. word_counter
   Use to count words in text.

5. note_manager
   Use to save, read, or append text notes.

Rules:

- Use the correct tool whenever the request matches
  one of the available tools.

- Never invent missing tool parameters.

- If required information is missing, ask the user
  for it before calling the tool.

- Never estimate Titanic fares yourself.
  Always use titanic_fare_predictor.

- The Titanic tool predicts fare only.
  It does not provide historical Titanic facts.

- Use calculator for arithmetic instead of solving
  calculations manually.

- If the user asks for multiple supported tasks,
  complete all of them.

- Keep final answers clear, concise, natural,
  and human-readable.

- Do not expose internal JSON or implementation
  details in the final answer.
"""


# =========================================================
# Gemini Model
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME
)


# =========================================================
# Exactly 5 Tools
# =========================================================

tools = [
    calculator,
    titanic_fare_predictor,
    distance_converter,
    word_counter,
    note_manager,
]


# Bind tools to Gemini

llm_with_tools = llm.bind_tools(
    tools
)


# =========================================================
# LangGraph Node 1: Assistant
# =========================================================

def assistant_node(
    state: MessagesState
):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# =========================================================
# LangGraph Node 2: Tools
# =========================================================

tool_node = ToolNode(
    tools
)


# =========================================================
# Build Runtime Graph
# =========================================================

builder = StateGraph(
    MessagesState
)


builder.add_node(
    "assistant",
    assistant_node,
)


builder.add_node(
    "tools",
    tool_node,
)


# START -> Assistant

builder.add_edge(
    START,
    "assistant",
)


# Assistant decides:
#
# tool needed -> tools
# no tool      -> END

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)


# Tool execution -> Assistant again

builder.add_edge(
    "tools",
    "assistant",
)


# Compile runtime graph

graph = builder.compile()


# =========================================================
# Graph Visualization
# =========================================================

def visual_node(state):
    return {}


def validate_route(state):
    return "analyze_task"


def tool_route(state):
    return "calculator"


display_builder = StateGraph(
    MessagesState
)


# ---------------------------------------------------------
# Visualization Nodes
# ---------------------------------------------------------

display_builder.add_node(
    "validate_input",
    visual_node,
)

display_builder.add_node(
    "analyze_task",
    visual_node,
)

display_builder.add_node(
    "calculator",
    visual_node,
)

display_builder.add_node(
    "titanic_fare_predictor",
    visual_node,
)

display_builder.add_node(
    "distance_converter",
    visual_node,
)

display_builder.add_node(
    "word_counter",
    visual_node,
)

display_builder.add_node(
    "note_manager",
    visual_node,
)

display_builder.add_node(
    "format_result",
    visual_node,
)


# ---------------------------------------------------------
# START -> Validate Input
# ---------------------------------------------------------

display_builder.add_edge(
    START,
    "validate_input",
)


# ---------------------------------------------------------
# Validate Input -> Analyze Task
# ---------------------------------------------------------

display_builder.add_conditional_edges(
    "validate_input",
    validate_route,
    {
        "analyze_task": "analyze_task",
    },
)


# ---------------------------------------------------------
# Analyze Task -> Tool
# ---------------------------------------------------------

display_builder.add_conditional_edges(
    "analyze_task",
    tool_route,
    {
        "calculator": "calculator",
        "titanic_fare_predictor": "titanic_fare_predictor",
        "distance_converter": "distance_converter",
        "word_counter": "word_counter",
        "note_manager": "note_manager",
    },
)


# ---------------------------------------------------------
# Tools -> Format Result
# ---------------------------------------------------------

display_builder.add_edge(
    "calculator",
    "format_result",
)

display_builder.add_edge(
    "titanic_fare_predictor",
    "format_result",
)

display_builder.add_edge(
    "distance_converter",
    "format_result",
)

display_builder.add_edge(
    "word_counter",
    "format_result",
)

display_builder.add_edge(
    "note_manager",
    "format_result",
)


# ---------------------------------------------------------
# Format Result -> END
# ---------------------------------------------------------

display_builder.set_finish_point(
    "format_result"
)


# Compile visualization graph

display_graph = display_builder.compile()


# =========================================================
# Human-readable AI response
# =========================================================

def extract_text(message):

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        parts.append(text)

            elif isinstance(item, str):

                parts.append(item)

        return "\n".join(
            parts
        ).strip()

    return str(content)


# =========================================================
# Clean Tool Trace
# =========================================================

def show_tool_trace(messages):

    for message in messages:

        # ---------------------------------------------
        # Tool requested by AI
        # ---------------------------------------------

        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
        ):

            for tool_call in message.tool_calls:

                tool_name = tool_call[
                    "name"
                ]

                arguments = tool_call[
                    "args"
                ]

                readable_args = ", ".join(
                    f"{key}={value!r}"
                    for key, value
                    in arguments.items()
                )

                print(
                    f"\n[Tool] {tool_name}"
                )

                if readable_args:

                    print(
                        f"[Input] {readable_args}"
                    )

        # ---------------------------------------------
        # Tool execution result
        # ---------------------------------------------

        if (
            type(message).__name__
            == "ToolMessage"
        ):

            print(
                f"[Result] {message.content}"
            )


# =========================================================
# Display Tools
# =========================================================

def show_tools():

    print("\nAvailable Tools")
    print("-" * 50)

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
        " - Save, read, append notes"
    )


# =========================================================
# Main Interactive Program
# =========================================================

def main():

    print(
        "\n"
        + "=" * 55
    )

    print(
        "LANGGRAPH AI AGENT"
    )

    print(
        "=" * 55
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Tools: {len(tools)}"
    )


    # =====================================================
    # Display Graph
    # =====================================================

    print("\nGraph Flow:")

    display_graph.get_graph().print_ascii()


    print(
        "\nCommands:"
        "\n  tools  - show available tools"
        "\n  clear  - clear conversation"
        "\n  exit   - close program"
    )

    print(
        "=" * 55
    )


    # -----------------------------------------------------
    # Conversation state
    # -----------------------------------------------------

    conversation = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]


    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()


            if not user_input:
                continue


            # =============================================
            # Exit
            # =============================================

            if user_input.lower() in [
                "exit",
                "quit",
                "bye",
            ]:

                print(
                    "\nAgent: Goodbye."
                )

                break


            # =============================================
            # Show tools
            # =============================================

            if user_input.lower() == "tools":

                show_tools()

                continue


            # =============================================
            # Clear conversation
            # =============================================

            if user_input.lower() == "clear":

                conversation = [
                    SystemMessage(
                        content=SYSTEM_PROMPT
                    )
                ]

                print(
                    "\nAgent: Conversation cleared."
                )

                continue


            # =============================================
            # Add User Message
            # =============================================

            conversation.append(
                HumanMessage(
                    content=user_input
                )
            )


            previous_count = len(
                conversation
            )


            # =============================================
            # Run LangGraph
            # =============================================

            result = graph.invoke(
                {
                    "messages":
                    conversation
                }
            )


            updated_messages = result[
                "messages"
            ]


            # Only messages created
            # during current graph execution

            new_messages = (
                updated_messages[
                    previous_count:
                ]
            )


            # =============================================
            # Tool Trace
            # =============================================

            if SHOW_TOOL_TRACE:

                show_tool_trace(
                    new_messages
                )


            # =============================================
            # Final Human-readable Answer
            # =============================================

            final_message = (
                updated_messages[-1]
            )


            answer = extract_text(
                final_message
            )


            print(
                f"\nAgent: {answer}"
            )


            # =============================================
            # Preserve Conversation State
            # =============================================

            conversation = (
                updated_messages
            )


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
# Run Program
# =========================================================

if __name__ == "__main__":
    main()