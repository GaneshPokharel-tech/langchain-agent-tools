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
LangChain Agent

The LangChain version is located in:

main.py

Run it with:

uv run main.py

Flow:

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
LangGraph Agent

The LangGraph version is located in:

langgraph_main.py

Run it with:

uv run langgraph_main.py

Graph flow:

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

The LangGraph version explicitly uses:

StateGraph
MessagesState
Assistant Node
ToolNode
Conditional Edges
Tool Routing
Setup

Install project dependencies:

uv sync

Create a .env file in the project folder:

GOOGLE_API_KEY=your_google_api_key

Do not upload the .env file to GitHub.

Titanic Fare Model

The Titanic tool uses a Linear Regression model trained on Titanic passenger data.

The model uses these features:

Pclass
Sex
Age
Embarked
FamilySize
IsAlone

The model predicts passenger fare.

The trained model is stored in:

titanic_fare_regression_model.joblib
Interactive Commands

Inside the agent:

tools

Shows all available tools.

clear

Clears the current conversation memory.

exit

Closes the program.

Example

User:

Calculate 50 + 34 / 45

Agent selects:

calculator

Tool result:

50.75555555555555

Final response:

The result is approximately 50.76.

Titanic example:

Predict Titanic fare for class 2, male, age 25,
embarked S, family size 1.

The agent automatically selects:

titanic_fare_predictor

and returns the predicted fare.

Technologies Used
Python
LangChain
LangGraph
Google Gemini
pandas
scikit-learn
joblib
python-dotenv
numexpr
uv
Purpose

The purpose of this project is to demonstrate how an LLM can interact with external tools instead of relying only on its own generated responses.

The project also demonstrates the difference between:

LangChain agent-based tool calling
LangGraph graph-based tool routing and execution