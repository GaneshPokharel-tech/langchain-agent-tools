import re
import numexpr as ne

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Supports:
    +, -, *, /, //, %, **, parentheses
    and functions such as sqrt().

    Examples:
    (25 + 15) * 3 / 2
    2 ** 8
    sqrt(144)
    """

    expression = expression.strip()

    if not expression:
        return "Error: empty expression."

    # Basic safety check
    allowed_pattern = r"^[0-9A-Za-z_+\-*/%.(),\s]+$"

    if not re.match(
        allowed_pattern,
        expression,
    ):
        return (
            "Error: expression contains "
            "unsupported characters."
        )

    try:

        result = ne.evaluate(
            expression
        )

        # Convert NumPy scalar/array result
        # into a normal Python value
        if hasattr(result, "item"):
            result = result.item()

        return str(result)

    except Exception as error:

        return (
            f"Calculator error: {error}"
        )


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    tests = [
        "(25 + 15) * 3 / 2",
        "2 ** 8",
        "100 / 4 + 10",
        "sqrt(144)",
    ]

    for expression in tests:

        print(
            f"\nExpression: {expression}"
        )

        result = calculator.invoke(
            {
                "expression": expression
            }
        )

        print(
            f"Result: {result}"
        )