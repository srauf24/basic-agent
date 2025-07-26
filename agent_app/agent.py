import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# Dummy function
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: The complete answer to the original input question, incorporating the information from the Observation.

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    
]

output = client.chat.completions.create( # This is the function you call to start a new chat interaction.
    messages=messages,  #The actual conversation history
    stream=False,
    max_tokens=200,
    stop=["Observation:"] # Let's stop before any actual function is called

)
print(f"Output of first call: {output.choices[0].message.content}")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    {"role": "assistant", "content": output.choices[0].message.content + "Observation: " + get_weather('London')}
]

output = client.chat.completions.create( # This is the function you call to start a new chat interaction.
    messages= messages,  #The actual conversation history
    stream=False,
    max_tokens=200,
    # complete execution
)
print(f"output of second call: ", output.choices[0].message.content)


