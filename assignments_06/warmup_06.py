from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI
from pypdf import PdfReader
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

#-------------------------------------------------------------------------------------------------
# --- RAG Concepts ---

# Concepts Q1
"""
Scenario A:
Best approach: RAG.

Reason:
The legal team needs answers based on a large internal policy library.
Because the PDFs are updated every quarter, RAG is better than fine-tuning because it can retrieve the most current documents without retraining the model.

Scenario B:
Best approach: Fine-tuning.

Reason:
The startup wants the model to consistently write in a specific brand voice.
Because they have 3,000 examples written by their own team, fine-tuning is a good choice for teaching the model that style.

Scenario C:
Best approach: Prompt engineering.

Reason:
The analyst only needs to ask questions about one short two-page report.
For a single small document, it is simplest to put the report directly into the prompt instead of building a RAG system or fine-tuning a model.
"""

print("\n--- RAG Concepts ---")
print("Concepts Q1 completed. See comments in the code.")

# Concepts Q2
"""
A confidently wrong answer is more harmful than an answer that says "I am not sure"
because people are more likely to trust information that sounds clear, complete,
and certain. When a model admits uncertainty, the user knows they should verify
the answer before acting on it.

For example, if an AI medical assistant confidently gives the wrong dosage for a
medication, a patient might follow that advice and get hurt. The confident tone
makes the answer feel reliable even though the content is incorrect.
"""

print("Concepts Q2 completed. See comments in the code.")

# Concepts Q3
"""
Correct order of a complete RAG pipeline:

1. Extract text from source documents
   The system reads the original documents and pulls out the text that can be searched.

2. Split text into chunks
   The long document text is divided into smaller pieces so the system can retrieve only the most relevant sections.

3. Convert text chunks into embeddings
   Each chunk is converted into a numeric vector that represents its meaning.

4. Receive the user's query
   The system receives the question or request from the user.

5. Embed the user's query
   The user's query is converted into an embedding so it can be compared with the document chunks.

6. Retrieve the most relevant chunks
   The system searches for the chunks whose embeddings are most similar to the query embedding.

7. Inject retrieved chunks into the prompt
   The retrieved text is added to the prompt as context for the language model.

8. Generate a response from the LLM
   The LLM writes an answer using the user's query and the retrieved document context.
"""

print("Concepts Q3 completed. See comments in the code.")

#-------------------------------------------------------------------------------------------------
# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1
query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

keyword_q1_results = simple_keyword_retrieval(query, documents, verbose=True)
selected_document_name = keyword_q1_results[0][0]

print("\n--- Keyword RAG ---")
print(f"Keyword Q1 selected document: {selected_document_name}")

# Keyword Q1 explanation:
# The selected document is loyalty.txt because the simple keyword retriever found
# one overlapping token: "your". The more relevant document should be hours.txt,
# but the retriever did not match "weekend" from the query with "weekends" in
# hours.txt, which shows a limitation of basic keyword matching.

# Keyword Q2
query = "Do you have anything without caffeine?"

keyword_q2_results = simple_keyword_retrieval(query, documents, verbose=True)
selected_document_name = keyword_q2_results[0][0]

print(f"Keyword Q2 selected document: {selected_document_name}")

# Keyword Q2 explanation:
# The selected document is None found because there are no overlapping keywords
# between the query and the documents. Keyword RAG did not get this right because
# the menu document is the closest useful document, but it does not use the word
# "caffeine". A semantic retriever using embeddings would do better because it
# could understand that caffeine is related to coffee, espresso, lattes, and cold brew.

# Keyword Q3
# Prediction:
# I predict that loyalty.txt will be selected because the query asks about
# "rewards", and the loyalty program is the closest match in meaning. However,
# I also know that the exact word "rewards" does not appear in the document,
# so the simple keyword retriever might fail or select something unexpected.

query = "How do I sign up for rewards?"

keyword_q3_results = simple_keyword_retrieval(query, documents, verbose=True)
selected_document_name = keyword_q3_results[0][0]

print(f"Keyword Q3 selected document: {selected_document_name}")

# Keyword Q3 result:
# My prediction was not correct. I expected loyalty.txt because "rewards" is
# semantically related to a loyalty program, but the retriever selected None found.
# This happened because the document says "loyalty program", "earn", "points",
# and "redeem", but it never uses the exact word "rewards". This shows that
# simple keyword retrieval cannot understand synonyms or related concepts.

#-------------------------------------------------------------------------------------------------
# --- Semantic RAG Concepts ---

# Semantic Q1
"""
A vector embedding is a list of numbers that represents the meaning of a piece
of text. Texts with similar meanings should have embeddings that are close to
each other in vector space.

The chunk with a cosine similarity score of 0.85 is more relevant than the chunk
with a score of 0.30. A higher cosine similarity means the query and the chunk
are closer in meaning, so 0.85 suggests a strong semantic relationship while
0.30 suggests a weaker relationship.

Semantic search can find a relevant chunk even when the exact words do not match
because embeddings represent meaning, not just literal keywords. For example,
a query about "rewards" could match a chunk about a "loyalty program" because
those ideas are related.
"""

print("\n--- Semantic RAG Concepts ---")
print("Semantic Q1 completed. See comments in the code.")

# Semantic Q2
"""
| Feature                    | Keyword RAG                    | Semantic RAG                            |
|----------------------------|--------------------------------|-----------------------------------------|
| What is compared?          | Exact word overlap             | Vector embeddings / meaning similarity  |
| What is retrieved?         | Full document                  | Most relevant chunks                    |
| Can it handle synonyms?    | No                             | Yes, usually                            |
| Storage format             | Plain text dictionary          | Vector store / embedding index          |
| Relevance score            | Number of overlapping keywords | Cosine similarity score                 |
"""

print("Semantic Q2 completed. See comments in the code.")

#-------------------------------------------------------------------------------------------------
# --- LlamaIndex ---

# LlamaIndex Q1
possible_pdf_directories = [
    "assignments_06/brightleaf_pdfs",
    "../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs",
    "lessons/06_AI_augmentation/resources/brightleaf_pdfs",
]

pdf_directory = None

for possible_directory in possible_pdf_directories:
    if os.path.isdir(possible_directory):
        pdf_directory = possible_directory
        break

if pdf_directory is None:
    raise FileNotFoundError(
        "Could not find the brightleaf_pdfs directory. "
        "Check that the lesson materials are available locally."
    )

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

print("\n--- LlamaIndex ---")


def load_pdf_documents(folder_path):
    """Load PDF files with pypdf and convert them into LlamaIndex Documents."""
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)

            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)

            full_text = "\n".join(pages_text)

            documents.append(
                Document(
                    text=full_text,
                    metadata={"filename": filename},
                )
            )

    return documents


documents = load_pdf_documents(pdf_directory)

print(f"Loaded {len(documents)} PDF documents.")

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
)

retriever = index.as_retriever(similarity_top_k=3)

client = OpenAI()

for question in questions:
    print(f"\nQuestion: {question}")

    retrieved_nodes = retriever.retrieve(question)

    context_chunks = []
    for node_number, node in enumerate(retrieved_nodes, start=1):
        chunk_text = node.node.get_content()
        filename = node.node.metadata.get("filename", "Unknown file")
        context_chunks.append(chunk_text)

        print(f"\nSource node {node_number}")
        print(f"Source file: {filename}")
        print(f"Similarity score: {node.score}")
        print(f"Chunk preview: {chunk_text[:150]}")

    context = "\n\n".join(context_chunks)

    prompt = f"""
Use the context below to answer the question. If the context does not contain
the answer, say that the answer is not available in the provided context.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers using only the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    print("\nAnswer from the model:")
    print(answer)


# LlamaIndex Q1 observations:
"""
Query 1: What employee benefits does BrightLeaf offer?

The retrieved chunks looked mostly relevant. The top source node came from
employee_benefits.pdf, which is the correct and most useful document for this
question. The second and third nodes came from partnerships.pdf and
mission_statement.pdf, which were less directly relevant.

The model's response sounded confident and specific. It listed concrete benefits
such as medical insurance, vision benefits, wellness programs, life insurance,
disability insurance, a 401(k) match, parental leave, flexible work options,
professional development, mentorship, DEI support, and online courses.

The unexpected retrievals were partnerships.pdf and mission_statement.pdf. They
are related to BrightLeaf as a company, but they are not the best sources for
employee benefits.


Query 2: What are BrightLeaf's security policies?

The retrieved chunks looked mostly relevant. The top source node came from
security_policy.pdf, which is the correct and most useful document for this
question. The second and third nodes came from employee_benefits.pdf and
mission_statement.pdf, which were less directly relevant.

The model's response sounded confident and specific. It described network and
data security, incident response, employee training, access governance, vendor
security, hardware security, compliance, and governance.

The unexpected retrievals were employee_benefits.pdf and mission_statement.pdf.
They were not harmful because the top retrieved chunk contained the relevant
security policy text, but they show that semantic retrieval can still bring in
company-related documents that are only loosely related to the question.
"""

# LlamaIndex Q2
print("\n--- LlamaIndex Q2 ---")

comparison_question = "What are BrightLeaf's security policies?"

for top_k in [1, 5]:
    print(f"\nRunning query with similarity_top_k={top_k}")
    print(f"Question: {comparison_question}")

    comparison_retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = comparison_retriever.retrieve(comparison_question)

    context_chunks = []
    for node_number, node in enumerate(retrieved_nodes, start=1):
        chunk_text = node.node.get_content()
        filename = node.node.metadata.get("filename", "Unknown file")
        context_chunks.append(chunk_text)

        print(f"\nSource node {node_number}")
        print(f"Source file: {filename}")
        print(f"Similarity score: {node.score}")

    context = "\n\n".join(context_chunks)

    prompt = f"""
Use the context below to answer the question. If the context does not contain
the answer, say that the answer is not available in the provided context.

Context:
{context}

Question:
{comparison_question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers using only the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    print("\nResponse:")
    print(answer)


# LlamaIndex Q2 observations:
"""
I compared the same security policy query with similarity_top_k=1 and
similarity_top_k=5.

With similarity_top_k=1, the retriever returned only security_policy.pdf. This
was enough for the model to produce a detailed and specific answer because the
top source node contained the main security policy information.

With similarity_top_k=5, the retriever returned security_policy.pdf plus
employee_benefits.pdf, mission_statement.pdf, partnerships.pdf, and
earnings_report.pdf. The model's answer did not change much because the first
source node already had the relevant information.

More retrieved context is not always better. It can help when the answer is
spread across several chunks, but it can also add noise when extra chunks come
from documents that are only loosely related to the question.
"""

# LlamaIndex Q3
print("\n--- LlamaIndex Q3 ---")

struggle_question = "Should I invest in BrightLeaf Solar?"

print(f"\nQuestion: {struggle_question}")

struggle_retriever = index.as_retriever(similarity_top_k=3)
retrieved_nodes = struggle_retriever.retrieve(struggle_question)

context_chunks = []

for node_number, node in enumerate(retrieved_nodes, start=1):
    chunk_text = node.node.get_content()
    filename = node.node.metadata.get("filename", "Unknown file")
    context_chunks.append(chunk_text)

    print(f"\nSource node {node_number}")
    print(f"Source file: {filename}")
    print(f"Similarity score: {node.score}")
    print(f"Chunk text:")
    print(chunk_text)

context = "\n\n".join(context_chunks)

prompt = f"""
Use the context below to answer the question. If the context does not contain
enough information to answer safely, say what information is missing.

Context:
{context}

Question:
{struggle_question}
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers using only the provided context.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

answer = response.choices[0].message.content

print("\nResponse:")
print(answer)


# LlamaIndex Q3 observations:
"""
I expected this query to be difficult because "Should I invest in BrightLeaf
Solar?" asks for a recommendation, not just a factual answer from one document.

The retriever returned mission_statement.pdf, partnerships.pdf, and
earnings_report.pdf. This made sense because the question requires information
about the company's mission, growth strategy, partnerships, financial
performance, risks, and future outlook.

The model gave a fairly careful answer. It summarized positive signals such as
revenue growth, partnerships, expansion plans, and community impact, but it also
noted missing information such as current stock price, valuation, competitive
position, dividend policy, liquidity, macroeconomic factors, and personal
investment goals.

I would improve this system by making the prompt stricter for financial or
recommendation-style questions. The model should clearly separate facts from
investment advice, avoid giving a direct recommendation, list missing evidence,
and suggest what additional documents or data would be needed before making a
decision.
"""

# LlamaIndex Q4
print("\n--- LlamaIndex Q4 ---")

judge_llm = LlamaOpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    llm=LlamaOpenAI(model="gpt-4o-mini"),
)

evaluation_queries = [
    "What employee benefits does BrightLeaf offer?",
    "What is BrightLeaf's cafeteria lunch menu?",
]

for eval_query in evaluation_queries:
    print(f"\nEvaluation query: {eval_query}")

    response = query_engine.query(eval_query)

    print("\nResponse:")
    print(response)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=eval_query,
        response=response,
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=eval_query,
        response=response,
    )

    print("\nFaithfulness score:")
    print(faithfulness_result.score)
    print("Faithfulness passing:")
    print(faithfulness_result.passing)
    print("Faithfulness feedback:")
    print(faithfulness_result.feedback)

    print("\nRelevancy score:")
    print(relevancy_result.score)
    print("Relevancy passing:")
    print(relevancy_result.passing)
    print("Relevancy feedback:")
    print(relevancy_result.feedback)


# LlamaIndex Q4 observations:
"""
A faithfulness score of 1.0 means the response is supported by the retrieved
context. A score of 0.0 would indicate that the response contains claims that
are not supported by the retrieved context, which could mean the model
hallucinated or added outside information.

A relevancy score measures whether the response actually answers the user's
query. This is different from faithfulness because an answer can be faithful to
the context but still not be relevant to the specific question.

The scores changed between the two queries. The employee benefits query received
faithfulness 1.0 and relevancy 1.0 because the BrightLeaf documents contain
direct information about employee benefits, and the response answered the
question with supported details.

The cafeteria lunch menu query received faithfulness 1.0 but relevancy 0.0.
This makes sense because the model did not hallucinate a cafeteria menu; it
truthfully said that the information was not provided. However, it still did not
answer the user's requested question with menu details, so the relevancy score
was low.

The LLM-as-a-judge approach uses another language model to evaluate the RAG
response. It is used because RAG answers are often open-ended, so there is not
always one exact string that can be compared with a simple accuracy metric. The
judge model can evaluate qualities like groundedness, relevance, and whether
the response is supported by the retrieved context.
"""
