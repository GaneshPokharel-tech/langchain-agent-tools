from pathlib import Path

from langchain_core.tools import tool


NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)


@tool
def note_manager(
    action: str,
    filename: str,
    content: str = "",
) -> str:
    """
    Manage text notes.

    Actions:
    save   - create or overwrite a note
    read   - read a note
    append - add text to an existing note

    filename example:
    study.txt
    """

    action = action.lower().strip()
    filename = filename.strip()

    if not filename:
        return "Error: filename is required."

    if not filename.endswith(".txt"):
        filename += ".txt"

    # Prevent paths such as ../../file
    safe_filename = Path(filename).name
    file_path = NOTES_DIR / safe_filename

    if action == "save":

        if not content.strip():
            return "Error: content is required for save."

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return f"Note saved: {safe_filename}"

    elif action == "read":

        if not file_path.exists():
            return f"Error: {safe_filename} does not exist."

        note = file_path.read_text(
            encoding="utf-8"
        )

        return note

    elif action == "append":

        if not content.strip():
            return "Error: content is required for append."

        with file_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            if file_path.stat().st_size > 0:
                file.write("\n")

            file.write(content)

        return f"Note updated: {safe_filename}"

    else:

        return (
            "Error: action must be "
            "save, read, or append."
        )


if __name__ == "__main__":

    print(
        note_manager.invoke(
            {
                "action": "save",
                "filename": "agent_notes.txt",
                "content": "LangChain agents can use tools.",
            }
        )
    )

    print(
        note_manager.invoke(
            {
                "action": "append",
                "filename": "agent_notes.txt",
                "content": "This is our fifth tool.",
            }
        )
    )

    print("\nSaved note:")

    print(
        note_manager.invoke(
            {
                "action": "read",
                "filename": "agent_notes.txt",
            }
        )
    )