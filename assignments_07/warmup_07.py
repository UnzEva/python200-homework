import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()

possible_resources_dirs = [
    Path("assignments_07/resources"),
    Path("../python-200/lessons/07_AI_agents/resources"),
    Path("lessons/07_AI_agents/resources"),
]

RESOURCES_DIR = None

for possible_dir in possible_resources_dirs:
    if possible_dir.exists():
        RESOURCES_DIR = possible_dir
        break

assert RESOURCES_DIR is not None, "Could not find the resources directory."

print(f"Using resources directory: {RESOURCES_DIR}")

# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1
# ----------------------------------------------------------------------
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in degrees Celsius.",
            },
        },
        "required": ["celsius"],
    },
}

print("--- Lesson 02: Tool Definitions and the ReAct Loop ---")
print("Q1: Direct function calls")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

print("\nQ1: JSON schema")
print(celsius_to_fahrenheit_schema)

# Q2
# ------------------------------------------------------------------------------------
"""
Prediction:
Calling run_agent("Convert 100 degrees Celsius to Fahrenheit") will probably
not trigger a tool call because the only available tool is get_current_time.
That tool can answer questions about the current time, but it cannot convert temperatures.

I expect one API call. The model should answer directly from its own reasoning
instead of calling the time tool and then making a second API call.
"""
def get_current_time() -> str:
    """Return the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def run_time_only_agent(user_message: str) -> str:
    """Run a simple agent with get_current_time as its only tool."""
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools only when they are relevant.",
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[get_current_time_schema],
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    if first_message.tool_calls:
        print("Tool call requested.")
        messages.append(first_message)

        for tool_call in first_message.tool_calls:
            print(f"Tool called: {tool_call.function.name}")
            if tool_call.function.name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = "Unknown tool."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[get_current_time_schema],
            tool_choice="auto",
        )

        return second_response.choices[0].message.content

    return first_message.content


print("\nQ2: Agent with get_current_time as the only tool")
q2_result = run_time_only_agent("Convert 100 degrees Celsius to Fahrenheit")
print(q2_result)

# Q2 result:
# My prediction was correct. The model answered directly without using the get_current_time tool. 
# The temperature conversion did not require the current time, so the tool was irrelevant and only one API call was needed.

# Q3
# ------------------------------------------------------------------------------------------

celsius_to_fahrenheit_tool_schema = {
    "type": "function",
    "function": celsius_to_fahrenheit_schema,
}

def run_agent(user_message: str) -> str:
    """Run a simple agent with get_current_time and celsius_to_fahrenheit tools."""
    client = OpenAI()

    tools = [
        get_current_time_schema,
        celsius_to_fahrenheit_tool_schema,
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools only when they are relevant.",
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    if first_message.tool_calls:
        print("Tool call requested.")
        messages.append(first_message)

        for tool_call in first_message.tool_calls:
            print(f"Tool called: {tool_call.function.name}")
            if tool_call.function.name == "get_current_time":
                tool_result = get_current_time()
            elif tool_call.function.name == "celsius_to_fahrenheit":
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(arguments["celsius"])
            else:
                tool_result = "Unknown tool."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        return second_response.choices[0].message.content

    return first_message.content

# Q3
print("\nQ3: Agent with get_current_time and celsius_to_fahrenheit tools")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Response A explanation:
# The celsius_to_fahrenheit tool was called because the user asked for a specific Celsius-to-Fahrenheit conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Response B explanation:
# No tool was called for this query. The model answered from general knowledge
# because the user asked for a plain-English explanation, not just a direct Celsius-to-Fahrenheit conversion.

# --- Lesson 03: Multi-Tool Agent ---

# Q4
# -----------------------------------------------------------------------------------------

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        missing = [column for column in [col1, col2] if column not in self.df.columns]
        if missing:
            return {"error": f"These columns are not in the data: {missing}"}

        pearson_r, p_value = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."


csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "compute_correlation": csv_backend.compute_correlation,
    "plot_data": csv_backend.plot_data,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns using pandas.describe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column using pandas.describe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "The first numeric column name.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "The second numeric column name.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
]


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one ReAct agent loop using a simple tool-using agent.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """Return a tool result as a message for the LLM conversation history."""
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tool_call.model_dump() for tool_call in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as error:
                    print(f"Tool error in {name}: {type(error).__name__}: {error}")
                    result = {"error": f"Tool '{name}' failed: {type(error).__name__}: {error}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work and do not guess. "
    "If no CSV is loaded yet, load one first or list available CSV files. "
    "Keep answers short and student-friendly."
)

q4_messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

print("\n--- Lesson 03: Multi-Tool Agent ---")
print("Q4: Correlation tool test")

q4_response = run_agent_cycle(
    q4_messages,
    "Load bike_commute.csv and compute the correlation between avg_speed_kmh and avg_heart_rate.",
)

print("Q4 response:")
print(q4_response)

# Q4 reflection:
# I added compute_correlation as a new CsvManager method, added its JSON schema entry to tools_schema, and added it to node_tools. 
# This gives the agent a real tool for correlation instead of forcing it to guess or hit the tool-round limit.

# Q5
# ------------------------------------------------------------------------------------------

print("\nQ5: Recreate the correlation scenario with the new tool")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)

print("Q5 response:")
print(result)

# Q5 reflection:
# With the new compute_correlation tool in place, the agent succeeded instead of hitting the tool-round limit. 
# It loaded bike_commute.csv, called compute_correlation for avg_traffic_density and avg_speed_kmh, 
# and returned a moderate negative correlation of about -0.53. 
# This result makes sense because higher traffic density should usually be connected with lower average speed.

# Q6
# ---------------------------------------------------------------------------------------

print("\nQ6: Full messages list after the ReAct loop")

# ReAct role explanation:
# system: Gives the agent its overall instructions and behavior rules.
# user: Represents the user's request or question.
# assistant: Represents the model's reasoning step; it either answers directly
# or requests one or more tool calls.
# tool: Represents the observation step; it stores the result returned by a
# Python function so the model can use that result in the next reasoning step.

print(json.dumps(messages, indent=2, default=str))
