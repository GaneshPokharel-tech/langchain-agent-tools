# LangChain and LangGraph Tool-Calling Agent

This project demonstrates an AI agent built using both LangChain and LangGraph.

The agent uses Google Gemini and can automatically choose and execute one of five tools based on the user's request.

## Features

- Interactive command-line AI assistant
- LangChain agent implementation
- LangGraph agent implementation
- Automatic tool selection
- Conversation memory
- Human-readable responses
- Tool execution trace
- Missing-input handling
- Machine learning model integration

## Available Tools

### 1. Calculator
Performs mathematical calculations.

### 2. Titanic Fare Predictor
Predicts Titanic passenger fare using a trained Linear Regression model.

Required inputs:

- Passenger class
- Sex
- Age
- Embarkation port
- Family size

### 3. Distance Converter
Converts:

- Kilometers to miles
- Miles to kilometers

### 4. Word Counter
Counts the number of words in a given text.

### 5. Note Manager
Can:

- Save notes
- Read notes
- Append text to notes

---

## Project Structure

```text
langchain-agent-tools/
├── main.py
├── langgraph_main.py
├── calculator_tool.py
├── titanic_tool.py
├── distance_tool.py
├── word_tool.py
├── note_tool.py
├── train_titanic_model.py
├── download_titanic.py
├── titanic.csv
├── titanic_fare_regression_model.joblib
├── notes/
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## LangChain Agent

The LangChain version is implemented in:

```text
main.py
```

Run it with:

```bash
uv run main.py
```

### LangChain Flow

```text
User
  ↓
LangChain Agent
  ↓
Gemini
  ↓
Tool Selection
  ↓
Tool Execution
  ↓
Final Answer
```

The LangChain agent automatically decides which tool to use based on the user's request.

---

## LangGraph Agent

The LangGraph version is implemented in:

```text
langgraph_main.py
```

Run it with:

```bash
uv run langgraph_main.py
```

### LangGraph Flow

```text
START
  ↓
Assistant Node
  ↓
Tool Needed?
 ├── No → END
 └── Yes
      ↓
   ToolNode
      ↓
   Assistant Node
      ↓
     END
```

The LangGraph implementation uses:

- `StateGraph`
- `MessagesState`
- Assistant Node
- `ToolNode`
- Conditional Edges
- Tool Routing
- Conversation State

---

## Setup

Install project dependencies:

```bash
uv sync
```

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_api_key
```

The `.env` file is ignored by Git and should never be uploaded to GitHub.

---

## Titanic Fare Model

The Titanic fare prediction tool uses a Linear Regression model trained on Titanic passenger data.

### Input Features

- `Pclass`
- `Sex`
- `Age`
- `Embarked`
- `FamilySize`
- `IsAlone`

The trained model is stored in:

```text
titanic_fare_regression_model.joblib
```

The model is loaded and used by:

```text
titanic_tool.py
```

---

## Interactive Commands

### Show Available Tools

```text
tools
```

Displays all five available tools.

### Clear Conversation

```text
clear
```

Clears the current conversation context.

### Exit

```text
exit
```

Closes the program.

---

## Example: Calculator

User input:

```text
Calculate 50 + 34 / 45
```

Agent selects:

```text
calculator
```

Tool result:

```text
50.75555555555555
```

Final response:

```text
The result is approximately 50.76.
```

---

## Example: Titanic Fare Prediction

User input:

```text
Predict Titanic fare for class 2, male, age 25,
embarked S, family size 1.
```

Agent selects:

```text
titanic_fare_predictor
```

Example execution:

```text
[Tool] titanic_fare_predictor
[Input] pclass=2, sex='male', age=25, embarked='S', family_size=1
[Result] Predicted Titanic fare: 11.35
```

Final response:

```text
The predicted Titanic fare is $11.35.
```

---

## Missing Information Handling

If the user asks:

```text
What is the Titanic fare?
```

The agent does not invent missing values.

Instead, it asks for:

- Passenger class
- Sex
- Age
- Embarkation port
- Family size

After the user provides the missing information, the agent calls the Titanic fare prediction tool.

---

## Technologies Used

- Python
- LangChain
- LangGraph
- Google Gemini
- pandas
- scikit-learn
- joblib
- python-dotenv
- numexpr
- uv

---

## Project Purpose

This project demonstrates how an LLM can interact with external tools instead of relying only on generated responses.

It also compares two approaches to tool-calling agents.

### LangChain

LangChain provides a higher-level agent abstraction that automatically manages tool selection and execution.

### LangGraph

LangGraph provides explicit control over the workflow using:

- State
- Nodes
- Edges
- Conditional routing
- Tool execution

This makes the agent workflow easier to visualize, control, and extend.