"""
Week 10 Warmup: LLMs in Pipelines

This file contains conceptual answers as comments for using LLMs as transform
steps in data pipelines and for understanding Azure OpenAI.
"""


# --- LLMs as Transform ---


# Q1
"""
1. Parse the string "Jan 5th, 2024" into ISO date format:
I would use deterministic code because date parsing is a structured formatting
task that libraries like dateutil or pandas can handle reliably and cheaply.

2. Classify "my card was charged twice" as billing, technical, or general:
I would use an LLM because the task requires interpreting freeform language and
mapping it to a small set of business categories.

3. Calculate the average of a list of numbers:
I would use deterministic code because arithmetic should be exact, fast, cheap,
and reproducible.

4. Extract the company name from "Sr. Data Eng @ Acme Corp (contract)":
I would use an LLM because job titles can be messy and inconsistent, so the task
requires flexible text understanding.

5. Determine whether a product review is more than 100 words long:
I would use deterministic code because counting words is simple, cheap, and does
not require language understanding.
"""


# Q2
"""
A better prompt would keep the same task, but make the output structured and
pipeline-friendly:

Summarize the customer review in one or two sentences. Return only valid JSON in
this exact format:

{
  "summary": "short summary here"
}

Do not include markdown, extra text, or any fields other than summary.
"""


# Q3
"""
If the dataset has 50,000 records and each classification call takes 1 second,
sequential processing would take 50,000 seconds.

50,000 seconds is about 833 minutes, or about 13.9 hours.

One practical strategy is to process records concurrently in batches while
respecting rate limits. For example, the pipeline could use a worker pool or
async requests to classify multiple records at the same time, retry failed calls,
and write intermediate results so the whole job does not have to restart from
the beginning.
"""


# --- Azure OpenAI ---


# Q1
"""
A company might use Azure OpenAI instead of the regular OpenAI API because of
data residency, enterprise governance, and Azure integration.

Data residency means the company's API requests and data stay within Azure's
infrastructure instead of being sent directly to OpenAI's public API. This can be
important for regulated industries such as healthcare, finance, education, or
government.

Azure OpenAI can also fit better into existing enterprise controls, such as
Azure role-based access control, networking, logging, compliance policies, and
centralized billing or procurement.
"""


# Q2
"""
The main Azure-specific parameters when creating an AzureOpenAI client are
azure_endpoint and api_version.

azure_endpoint tells the SDK which Azure OpenAI resource to call, for example:
https://my-resource.openai.azure.com/

api_version tells the SDK which Azure OpenAI API version to use.

The deployment name is also required when making a request, but it is usually
passed as the model parameter in chat.completions.create(). Some SDK versions
also allow azure_deployment to be set at client initialization as a default, but
I would treat the deployment name separately from the core client setup.
"""


# Q3
"""
When using AzureOpenAI, the model parameter in chat.completions.create() takes
the Azure deployment name, not a public model name like "gpt-4o-mini".

The correct value is found in Azure AI Foundry or the Azure OpenAI resource under
the model deployments section. It is the deployment name created by the
organization, such as "gpt-4o-mini-prod" or another custom deployment name.
"""