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

failure_question = "What is the Wi-Fi password at Groundwork Coffee?"

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
I asked: "What is the Wi-Fi password at Groundwork Coffee?"

I expected this to be hard because it is a realistic customer question, but the
provided documents do not appear to contain the actual Wi-Fi password.

The retrieved source nodes came from our_story.txt, menu.txt, and faq.txt. These
documents are about the company's background, menu, hours, locations, and other
general information, but the retrieved previews did not include a Wi-Fi password.

The assistant did not invent a specific password, which is good. However, it
still guessed that the customer can ask a barista for the current password. That
sounds reasonable, but it is not directly supported by the retrieved documents.
This is a subtle failure: the model's answer sounds helpful and confident even
though the evidence is missing.

This suggests that AI-generated responses can sound trustworthy even when they
are filling in gaps. For a customer-facing assistant, I would want the model to
say something like, "I do not see the Wi-Fi password in the available documents."

To improve the system, I would use a stricter prompt that requires the assistant
to answer only from the retrieved context and explicitly say when information is
missing. I would also add a more complete FAQ document with Wi-Fi, parking,
accessibility, reservations, and other common customer questions.
"""

