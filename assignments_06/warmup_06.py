from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


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