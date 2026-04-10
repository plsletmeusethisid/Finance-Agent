import anthropic
from knowledge_base import search, get_stats
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """
You are a helpful Finance Agent for Shinwootns, a small cloud security company based in South Korea.

You assist employees with finance related questions based strictly on company finance documents provided to you.

Your areas of expertise include:
- Invoice processing and approval procedures
- Budget inquiries and expense tracking
- Expense approvals and reimbursement policies
- Financial reporting procedures
- Purchase request processes
- Allowances and company provided benefits (lunch allowance etc.)
- Vendor payment procedures

Rules:
- Only answer based on the document context provided to you
- Always cite which document your answer came from using [Source: filename]
- If the answer is not in the provided documents, say:
  "I couldn't find this information in the company finance documents.
   Please contact the Finance department directly."
- Never make up or infer information not explicitly in the documents
- Be precise, professional, and detail oriented in your responses
- Never share, confirm, or discuss specific salary information, individual compensation, or personal financial data
- For budget approvals or large expenditures, always direct to the appropriate manager
- If asked about financial figures not in the documents, decline and direct to the finance team
"""

def build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."

    context = "Relevant information retrieved from finance documents:\n\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"[{i}] Source: {chunk['source']}\n"
        context += f"{chunk['content']}\n\n"
    return context.strip()

def ask(question: str, conversation_history: list = None) -> tuple:
    if conversation_history is None:
        conversation_history = []

    print("  🔍 Searching finance documents...")
    chunks  = search(question, n_results=5)
    context = build_context(chunks)

    if chunks:
        sources = list(set(c["source"] for c in chunks))
        print(f"  📄 Found in: {', '.join(sources)}")
    else:
        print("  ⚠️  No relevant documents found")

    user_message = f"""Question: {question}

---
{context}
---

Please answer the question based only on the document context above.
Cite your sources."""

    messages = conversation_history + [
        {"role": "user", "content": user_message}
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = response.content[0].text

    updated_history = conversation_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history

def chat_loop():
    stats = get_stats()

    print("\n" + "=" * 60)
    print("💰  FINANCE AGENT — Shinwootns")
    print("=" * 60)

    if stats["total_chunks"] == 0:
        print("⚠️  Knowledge base is empty!")
        print("   Run: python sync_and_learn.py first\n")
        return

    print(f"📚 Knowledge base: {stats['total_chunks']} chunks loaded")
    print(f"🤖 Model: {MODEL}")
    print("Ask me anything about finance policies and procedures.")
    print("Type 'exit' to quit\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print()
        answer, history = ask(question, history)
        print(f"\n💰 Finance Agent: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    chat_loop()
