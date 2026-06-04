from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- Chat Completions API ---

# API Q1
#--------------------------------------------------------------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

print("Response text:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal tokens used:")
print(response.usage.total_tokens)

# API Q2
#--------------------------------------------------------------------------------
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )

    print(f"\nTemperature: {temp}")
    print("Response:")
    print(response.choices[0].message.content)

# In this run, the outputs did not differ very much across temperatures, but
# temperature still controls an important behavior. Mechanically, temperature
# changes how strongly the model favors the highest-probability next tokens.
# At temperature=0, the model is pushed toward the most likely token choices, so
# the output is more deterministic and repeatable. At higher temperatures, lower-
# probability tokens have more chance to be selected, which can make responses
# more varied, surprising, or creative.
#
# This naming prompt may not be the best test case for seeing temperature
# differences because short creative names often come from a small set of common
# branding patterns, and many possible answers are acceptable. A longer task, or
# running each temperature several times, would make the randomness easier to
# observe. If I needed a consistent, reproducible output, I would use
# temperature=0; if I wanted more brainstorming variety, I would try a higher
# temperature.   

# API Q3
#--------------------------------------------------------------------------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)

# Using n=3 returns multiple completions from a single API call.
# In this example, the responses are all correct and related, but they vary slightly in wording and focus. 
# This can be useful when you want several candidate responses and then choose the best one.

# API Q4
#--------------------------------------------------------------------------------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain how neural networks work."}
    ],
    max_tokens=15
)

print("\nAPI Q4 response:")
print(response.choices[0].message.content)

# The response was cut off because max_tokens=15 limited how many tokens the model was allowed to generate. 
# As a result, the answer stopped in the middle of a sentence instead of giving a complete explanation.
#
# In a real application, max_tokens is useful for controlling cost, latency, and output length. 
# It can help prevent overly long responses and keep the model focused when you only need a short answer.

# --- System Messages and Personas ---
# System Q1
#--------------------------------------------------------------------------------

messages_tutor = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response_tutor = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_tutor
)

print("\nTutor persona response:")
print(response_tutor.choices[0].message.content)


messages_interviewer = [
    {
        "role": "system",
        "content": "You are a strict technical interviewer. You answer briefly, directly, and without emotional encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response_interviewer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_interviewer
)

print("\nInterviewer persona response:")
print(response_interviewer.choices[0].message.content)

# The system message changed the tone, level of detail, and teaching style of the response.
# The tutor persona gave a longer, simpler, and more encouraging explanation with examples,
# while the interviewer persona was shorter, more direct, and less supportive.
# This shows that system messages are a powerful way to control how the model responds,
# even when the user asks the exact same question.

# System Q2
#--------------------------------------------------------------------------------

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 response:")
print(response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was included in the messages list sent with this API call. 
# The API itself is stateless, but the model can still use earlier messages in the same request as context.

# --- Prompt Engineering ---
# Prompt Q1 - Zero-Shot
#--------------------------------------------------------------------------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of this review as positive, negative, or mixed. "
                    "Respond with only one word: positive, negative, or mixed.\n\n"
                    f"Review: {review}"
                )
            }
        ]
    )

    print(f"\nReview {i}:")
    print(review)
    print("Sentiment:")
    print(response.choices[0].message.content)

# The zero-shot prompt worked well here because the task was simple and clearly stated.
# The model correctly identified positive, negative, and mixed sentiment without needing examples.

# Prompt Q2 - One-Shot
#--------------------------------------------------------------------------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
                    "Example:\n"
                    'Review: "Fast shipping but the item arrived damaged."\n'
                    "Sentiment: mixed\n\n"
                    f'Review: "{review}"\n'
                    "Sentiment:"
                )
            }
        ]
    )

    print(f"\nOne-shot Review {i}:")
    print(review)
    print("Sentiment:")
    print(response.choices[0].message.content)

# In this run, adding one example did not change the labels or the response format.
# The model was already consistent in Q1 and continued to return one-word answers:
# positive, negative, or mixed.
#
# Even though the output looked the same here, one-shot prompting can still be useful
# because it helps reinforce the desired format and may improve consistency on harder tasks.

# Prompt Q3 - Few-Shot
#--------------------------------------------------------------------------------

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

few_shot_examples = (
    'Review: "The staff was helpful and the whole experience felt easy."\n'
    "Sentiment: positive\n\n"
    'Review: "The app freezes every time I try to upload a file."\n'
    "Sentiment: negative\n\n"
    'Review: "The design is beautiful, but the battery life is disappointing."\n'
    "Sentiment: mixed\n\n"
)

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
                    f"{few_shot_examples}"
                    f'Review: "{review}"\n'
                    "Sentiment:"
                )
            }
        ]
    )

    print(f"\nFew-shot Review {i}:")
    print(review)
    print("Sentiment:")
    print(response.choices[0].message.content)

# In this example, zero-shot, one-shot, and few-shot prompting all produced the same correct labels. 
# That suggests the task was simple enough for the model to handle reliably even without examples.
#
# I would choose zero-shot when the task is straightforward and the instructions are already clear. 
# I would choose one-shot when I want to reinforce the desired output format with a single example. 
# I would choose few-shot when the task is more ambiguous or when I need stronger consistency in style, structure, or labeling.

# Prompt Q4 - Chain of Thought
#--------------------------------------------------------------------------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Solve the following problem. Show your reasoning step by step in plain text only, "
                "without LaTeX or special formatting, and clearly label the final answer.\n\n"
                "A data engineer earns $85,000 per year. She gets a 12% raise, "
                "then 6 months later takes a new job that pays $7,500 more per year "
                "than her post-raise salary. What is her final annual salary?"
            )
        }
    ]
)

print("\nPrompt Q4 response:")
print(response.choices[0].message.content)

# Asking the model to reason step by step can improve accuracy because it breaks
# a multi-step problem into smaller intermediate calculations instead of trying to jump straight to the answer. 
# That reduces the chance of skipping a step or combining the numbers incorrectly, especially in word problems with multiple changes.

# Prompt Q5 - Structured Output
#--------------------------------------------------------------------------------

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Analyze the following review and return the result only as valid JSON. "
                "Use exactly these keys: sentiment, confidence, reason. "
                "The confidence value must be a float from 0 to 1, and the reason must be one sentence.\n\n"
                f"Review: {review}"
            )
        }
    ]
)

raw_text = response.choices[0].message.content

cleaned_text = raw_text.strip()

if cleaned_text.startswith("```json"):
    cleaned_text = cleaned_text.removeprefix("```json").strip()

if cleaned_text.startswith("```"):
    cleaned_text = cleaned_text.removeprefix("```").strip()

if cleaned_text.endswith("```"):
    cleaned_text = cleaned_text.removesuffix("```").strip()

print("\nRaw response:")
print(raw_text)

try:
    parsed = json.loads(cleaned_text)

    print("\nParsed sentiment:")
    print(parsed["sentiment"])

    print("\nParsed confidence:")
    print(parsed["confidence"])

    print("\nParsed reason:")
    print(parsed["reason"])

except json.JSONDecodeError:
    print("\nFailed to parse valid JSON. Raw response for debugging:")
    print(raw_text)
    print("\nCleaned response for debugging:")
    print(cleaned_text)

# Structured output is useful because it makes the model response easier to parse and integrate into downstream code. 
# Returning valid JSON is especially helpful when an LLM is part of a larger automated pipeline.

# Prompt Q6 - Delimiters
#--------------------------------------------------------------------------------

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nDelimited instructions response:")
print(response.choices[0].message.content)


regular_text = "The park was quiet in the early morning, and a thin layer of fog hung above the grass."

second_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{regular_text}```
"""

second_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": second_prompt}]
)

print("\nDelimited prose response:")
print(second_response.choices[0].message.content)

# Delimiters help clearly separate the user's text from the instructions that tell the model what to do with that text. 
# This reduces the chance that the model will confuse the input content with the prompt itself, which improves reliability and
# makes the task easier to interpret correctly.

# --- Local Models with Ollama ---
# Ollama Q1
#--------------------------------------------------------------------------------

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\nOpenAI response:")
print(response.choices[0].message.content)

"""
Ollama output:
Thinking...
Okay, so the user wants me to explain what a large language model is in two sentences. Let me start by breaking down the key elements of a large language model. First, I know they're AI models, so I should mention 
that they're trained on massive datasets. Then, their primary function is to understand and generate human-like text. Maybe include examples like answering questions or creating stories. Wait, but how to make it 
concise in two sentences? Let me check if I'm including all necessary points without redundancy. Also, ensure the language is clear and straightforward. Maybe start with the definition, then the purpose, and then 
an example. That should cover it.
...done thinking.

A large language model is an AI system designed to understand and generate human-like text, such as answering questions or creating stories, by learning from vast datasets. It powers tools like chatbots, content 
generation, and language translation, enabling complex interactions with humans.
"""