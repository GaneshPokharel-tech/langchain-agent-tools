from langchain_core.tools import tool


@tool
def distance_converter(
    value: float,
    from_unit: str,
    to_unit: str,
) -> str:
    """
    Convert distance between kilometers and miles.

    Supported units:
    km
    miles
    """

    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    valid_units = ["km", "miles"]

    if from_unit not in valid_units:
        return "Error: from_unit must be km or miles."

    if to_unit not in valid_units:
        return "Error: to_unit must be km or miles."

    if value < 0:
        return "Error: distance cannot be negative."

    if from_unit == to_unit:
        result = value

    elif from_unit == "km" and to_unit == "miles":
        result = value * 0.621371

    elif from_unit == "miles" and to_unit == "km":
        result = value * 1.60934

    return f"{value} {from_unit} = {result:.4f} {to_unit}"


if __name__ == "__main__":

    result1 = distance_converter.invoke(
        {
            "value": 100,
            "from_unit": "km",
            "to_unit": "miles",
        }
    )

    result2 = distance_converter.invoke(
        {
            "value": 10,
            "from_unit": "miles",
            "to_unit": "km",
        }
    )

    print(result1)
    print(result2)
