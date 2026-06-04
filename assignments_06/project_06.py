from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- Step 1: Setup ---

possible_docs_dirs = [
    Path("assignments_06/resources/groundwork_docs"),
    Path("../python-200/lessons/06_AI_augmentation/resources/groundwork_docs"),
    Path("lessons/06_AI_augmentation/resources/groundwork_docs"),
]

docs_dir = None

for possible_dir in possible_docs_dirs:
    if possible_dir.exists():
        docs_dir = possible_dir
        break

assert docs_dir is not None, "Document directory not found. Check your Groundwork docs path."

print(f"Using document directory: {docs_dir}")


# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Loaded {len(documents)} documents.")

for document in documents:
    file_name = document.metadata.get("file_name", "Unknown file")
    print(f"Loaded document: {file_name}")

# --- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    llm=OpenAI(model="gpt-4o-mini"),
)

print("Index built successfully. Ready to answer questions.")    

# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n--- Step 4: Query the Assistant ---")

for question in questions:
    print(f"\nQuestion: {question}")

    response = query_engine.query(question)

    print("\nAnswer from the model:")
    print(response)

    source_nodes = response.source_nodes

    if source_nodes:
        top_node = source_nodes[0]
        file_name = top_node.node.metadata.get("file_name", "Unknown file")
        similarity_score = top_node.score
        chunk_text = top_node.node.get_content()

        print("\nTop retrieved source node:")
        print(f"Document name: {file_name}")
        print(f"Similarity score: {similarity_score}")
        print(f"Chunk preview: {chunk_text[:200]}")
    else:
        print("\nNo source nodes were retrieved.")


# Step 4 reflection:
"""
The assistant sounded confident and mostly accurate. It answered the questions
about weekend hours, dairy-free milk options, the loyalty program, Groundwork's
origin story, and catering or wholesale orders with specific details.

Some retrieved source nodes were exactly what I expected, such as our_story.txt
for the origin story and wholesale_catering.txt for catering and wholesale
orders. One surprising result was the weekend hours question: the answer was
correct, but the top retrieved source node was our_story.txt instead of faq.txt.
This shows why it is useful to inspect source nodes instead of trusting only the
final answer.

The dairy-free milk question also retrieved seasonal_specials.txt as the top
source. The answer still seemed accurate, but this result suggests that the
retriever may choose a related chunk even when another document might contain a
more direct answer.
"""

# --- Step 5: Find a Failure ---

failure_question = "Can I reserve the private room at Groundwork Coffee for a 40-person event next Friday?"

print("\n--- Step 5: Find a Failure ---")
print(f"\nQuestion: {failure_question}")

failure_response = query_engine.query(failure_question)

print("\nFull response:")
print(failure_response)

print("\nRetrieved source nodes:")

for node_number, source_node in enumerate(failure_response.source_nodes, start=1):
    file_name = source_node.node.metadata.get("file_name", "Unknown file")
    similarity_score = source_node.score
    chunk_text = source_node.node.get_content()

    print(f"\nSource node {node_number}")
    print(f"Document name: {file_name}")
    print(f"Similarity score: {similarity_score}")
    print(f"Chunk preview: {chunk_text[:200]}")


# Step 5 failure reflection:
"""
I asked: "Can I reserve the private room at Groundwork Coffee for a 40-person
event next Friday?"

I expected this to be difficult because the question combines information the
documents may partially cover with information they likely do not cover. The
documents mention catering or event-related services, but they do not provide a
reservation calendar, private room availability, or a policy for booking a
specific date.

This is a better failure case than asking for the Wi-Fi password because the FAQ
actually says customers should ask a barista for the current Wi-Fi password. In
that case, the model's answer was supported by the documents. Here, however, a
confident "yes" would go beyond the retrieved evidence.

A good answer should separate what is supported from what is missing. For
example, it could say that Groundwork offers catering for events of 20 people or
more if that appears in the retrieved context, but it should also say that the
documents do not confirm whether there is a private room or whether it is
available next Friday.

The actual response was better than a simple hallucination because it did not
claim that the room was available. However, it still could not fully answer the
user's real question because the retrieved documents do not include private room
availability or a reservation calendar.

This suggests that AI-generated responses can sound trustworthy even when they
are filling in gaps. For a customer-facing assistant, I would want the model to
say something like, "I do not see private room availability or reservation
details in the available documents."

To improve the system, I would use a stricter prompt that requires the assistant
to answer only from the retrieved context and explicitly say when information is
missing. I would also add a more complete FAQ or reservations document with room
availability, booking rules, event capacity, parking, accessibility, and other
common customer questions.
"""

# --- Step 6: Reflection ---

"""
The LlamaIndex version required about 8 lines for the core RAG setup: one line
to load the documents, four lines to build the vector index with an embedding
model, and three lines to create the query engine with similarity_top_k and the
LLM.

That is much shorter than the manual semantic RAG pipeline from the warmup,
which required roughly 40-50 lines to load files, create embeddings, compute
similarities, sort the results, assemble context, build a prompt, call the model,
and print source information.

The main advantage of LlamaIndex is that it packages the common RAG workflow into
a small, readable pipeline. The tradeoff is that some details are hidden, so it is
still important to inspect source nodes and understand what the query engine is
retrieving.
"""