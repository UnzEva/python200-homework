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
The prompt "Summarize this product review in a few sentences" creates a
downstream pipeline problem because the output is freeform text. The length,
format, wording, and structure may vary from one call to another, which makes it
hard to parse, validate, store in a table, or monitor for quality.

A better pipeline prompt would require a small, consistent JSON object:

System prompt:
You are a data transformation step in an ETL pipeline. Return only valid JSON.
Do not include markdown, commentary, or extra text.

User prompt:
Summarize the product review and return exactly this JSON schema:
{
  "summary": "one sentence summary of the review",
  "sentiment": "positive | neutral | negative",
  "main_issue": "short phrase or null"
}

Review:
<review text here>
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
An organization might use Azure OpenAI instead of the public OpenAI API because:

1. Enterprise governance and security:
Azure OpenAI can be managed inside the organization's Azure environment, using
Azure role-based access control, private networking options, monitoring, and
existing compliance processes.

2. Procurement and data residency requirements:
Many organizations already buy and manage cloud services through Azure. Azure
OpenAI lets them use OpenAI models through their existing Azure contracts,
billing, regional deployments, and enterprise controls.
"""


# Q2
"""
When switching from OpenAI to AzureOpenAI, the client initialization uses these
Azure-specific parameters:

1. azure_endpoint:
The URL of the organization's Azure OpenAI resource, such as
https://my-resource.openai.azure.com/.

2. api_version:
The Azure OpenAI API version to call. Azure OpenAI uses versioned API endpoints,
so the client must know which service API version to target.

3. azure_deployment:
The name of the deployed model inside the Azure OpenAI resource. This is the
deployment name chosen in Azure, not necessarily the base model name.
"""


# Q3
"""
When using AzureOpenAI, the model parameter in chat.completions.create() takes
the Azure deployment name, not a public model name like "gpt-4o-mini".

The correct value is found in Azure AI Foundry or the Azure OpenAI resource under
the model deployments section. It is the deployment name created by the
organization, such as "gpt-4o-mini-prod" or another custom deployment name.
"""