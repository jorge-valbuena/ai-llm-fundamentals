"""
Exercise: Nebula Cafe Chat Bot - RAG with Vector Database
This chatbot answers customer questions using only house policies stored in a vector DB.
Concepts used: RAG, Vector DB, Hallucination prevention, LLM Output control.
"""
import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI config
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

client = OpenAI(
    base_url=ENDPOINT_URL,
    api_key=AZURE_OPENAI_API_KEY
)

# --- Nebula Cafe Policies (the "knowledge base") ---
POLICIES = [
    "Nebula Cafe is open 7am-3pm on weekdays and 8am-1pm on weekends.",
    "We offer oat milk, almond milk, and soy milk at no extra charge with any drink.",
    "Customers receive a 50c discount on hot drinks when they bring a reusable cup.",
    "Dogs are allowed outside the cafe only. They are not permitted inside for health safety reasons.",
    "Free Wi-Fi is available, and the password can be found on your receipt after purchase.",
    "Table bookings at Nebula Cafe are only accepted for groups of 6 or more people.",
]

# --- Step 1: Store policies in ChromaDB (Vector Database) ---
vector_client = chromadb.Client()
collection = vector_client.get_or_create_collection(name="nebula_cafe_policies")

if collection.count() == 0:
    policy_ids = [f"policy_{i}" for i in range(len(POLICIES))]
    collection.add(documents=POLICIES, ids=policy_ids)


def retrieve_policy(question_text, top_k=1):
    """Find the most relevant policy for a question using vector similarity."""
    results = collection.query(
        query_texts=[question_text],
        n_results=top_k
    )
    return results["documents"][0][0] if results["documents"] else None


def answer_question(question, found_policy):
    """
    Ask the LLM to answer using ONLY the retrieved policy.
    This prevents hallucinations by grounding the response in a single policy.
    """
    prompt = f"""You are the Nebula Cafe customer service chatbot. 
Answer the customer's question using ONLY the following policy information provided below.

POLICY FOUND: {found_policy}

CUSTOMER QUESTION: {question}

Instructions:
- Use ONLY the POLICY FOUND to answer the question.
- If the policy does not contain enough information to answer, reply exactly with: "Sorry, I don't have that information."
- Do NOT make up or invent any details beyond what is in the policy.
- Keep your response short and friendly."""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def chat_bot():
    """Main chat loop: keep asking questions until the user types quit."""
    print("=" * 54)
    print("  Welcome to Nebula Cafe Chat Bot")
    print("  Ask about our policies, or type 'quit' to exit.")
    print("=" * 54)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("\\whatN0:smiling_face_with_horns: Chat Bot: Have a great day at Nebula Cafe!")
            break

        # Retrieve the best matching policy via vector search
        matched_policy = retrieve_policy(question)

        if not matched_policy:
            print(f"\nChat Bot: Sorry, I don't have that information.")
            continue

        # Show which policy was retrieved (for debugging / verification)
        print(f"\n[Retrieved Policy: {matched_policy}]")

        # Generate the LLM answer grounded in that single policy
        answer = answer_question(question, matched_policy)
        print(f"Chat Bot: {answer}")


if __name__ == "__main__":
    chat_bot()
