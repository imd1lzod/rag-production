import os
from typing import TypedDict, Annotated, Literal
from operator import add
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


class RAGState(TypedDict):

    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int


def create_vectorstore():

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    documents = [
        Document(
            page_content="""
        Opening UzCard and Humo plastic cards at our bank is completely free of charge. 
        Only a passport or ID card is required to issue a card. Cards are ready within 10-15 minutes. 
        Additionally, Visa and Mastercard cards can be opened for international payments.
        """,
            metadata={"source": "bank_cards_guide.md", "topic": "cards"},
        ),
        Document(
            page_content="""
        Microloans are provided to individuals in amounts up to 50 million sums, for a term of up to 36 months. 
        Applications are reviewed within 5 minutes via the mobile app, and funds are immediately credited to the card 
        upon approval. The annual interest rate starts from 24%.
        """,
            metadata={"source": "microloan_terms.md", "topic": "loans"},
        ),
        Document(
            page_content="""
        Mortgage loans are allocated for purchasing housing from the primary and secondary markets for up to 20 years. 
        The minimum down payment is 15%. Citizens of the Republic of Uzbekistan aged 21 to 65 with a regular official income 
        can apply.
        """,
            metadata={"source": "mortgage_policy.md", "topic": "mortgage"},
        ),
        Document(
            page_content="""
        National currency deposits yield returns of up to 22% per annum. Interest is paid out monthly or capitalized 
        by adding it to the principal amount. Even if a deposit is withdrawn early, a portion of the accrued interest 
        is retained.
        """,
            metadata={"source": "savings_accounts.md", "topic": "deposits"},
        ),
        Document(
            page_content="""
        All types of utility payments can be made commission-free through the mobile banking app. Additionally, 
        transfers between Humo and UzCard cards carry a 0.3% commission fee, and currency exchange transactions 
        operate at favorable rates.
        """,
            metadata={"source": "mobile_app_features.md", "topic": "mobile_banking"},
        ),
        Document(
            page_content="""
        Funds can be sent and received from abroad using international money transfer systems. Western Union, 
        Zolotaya Korona, and SWIFT transfers are available. There is an option to credit funds directly to national 
        or foreign currency cards through the mobile application.
        """,
            metadata={"source": "international_transfers.md", "topic": "transfers"},
        ),
        Document(
            page_content="""
        If a bank card is lost or its security is compromised, it is recommended to block or temporarily freeze it 
        immediately via the mobile app or the 24/7 call center at the short number 1234. The reissue fee is 30,000 sums.
        """,
            metadata={"source": "security_faq.md", "topic": "security"},
        ),
        Document(
            page_content="""
        Remote banking services are provided for business clients (sole proprietors and legal entities). A dedicated 
        internet banking system is available for opening accounts, managing payroll projects, and making tax payments. 
        Service fees depend on the selected tariff plan.
        """,
            metadata={"source": "business_banking.md", "topic": "business"},
        ),
        Document(
            page_content="""
        Under the cashback program, up to 1% cashback is returned for all types of terminal and online payments. Accumulated 
        cashback can be transferred to the main card for cash withdrawal or used for payments at any time. At special partner 
        stores, cashback reaches up to 5%.
        """,
            metadata={"source": "cashback_program.md", "topic": "loyalty"},
        ),
        Document(
            page_content="""
        Customer support (Call Center) operates 24 hours a day, 7 days a week. For any questions, you can call the short 
        number 1234 or get prompt answers from specialists via the online chat on the official website.
        """,
            metadata={"source": "customer_support.md", "topic": "support"},
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, collection_name="agentic-rag"
    )

    return vectorstore


def retrieve_documents(state: RAGState):

    query = state.get("rewritten_query") or state.get("query")

    if isinstance(query, list) and len(query) > 0:
        if isinstance(query[0], dict):
            query = query[0].get("text") or query[0].get("query", str(query[0]))
        else:
            query = str(query[0])
    elif isinstance(query, dict):
        query = query.get("text") or query.get("query", str(query))
    else:
        query = str(query)

    print(f"Searching: {query}")

    vectorstore = state.get("_vectorstore")
    if not vectorstore:
        vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    documents = retriever.invoke(query)

    print(f"Found {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(
            f" {i}. {doc.metadata.get("source", "unknown")}: {doc.page_content[:50]}..."
        )

    return {"documents": documents}


def grade_documents(state: RAGState):

    query = state["query"]
    documents = state["documents"]

    print(f"Evaluating {len(documents)} documents for relevance")

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a relevance grader. Given a user query and a document, determine if the document contains information relevant to answering the query.

Output ONLY a number between 0 and 1:
- 1.0 = Highly relevant, directly answers the query
- 0.7 = Somewhat relevant, contains related information
- 0.3 = Marginally relevant, tangentially related
- 0.0 = Not relevant at all

Output ONLY the number, nothing else.""",
            ),
            (
                "human",
                """Query: {query}

Document: {document}

Relevance score (0-1):""",
            ),
        ]
    )

    scores = []
    relevant_documents = []

    for doc in documents:
        chain = grading_prompt | llm
        result = chain.invoke({"query": query, "document": doc.page_content})

        try:
            score = float(result.content[0]["text"])
        except ValueError:
            score = 0.5

        scores.append(score)
        print(f"Score ---> {doc.page_content} --- {score:.2f}")

        if score >= 0.5:
            relevant_documents.append(doc)

    average_score = sum(scores) / len(scores) if scores else 0
    print(f"Average score --> {average_score:.2f}")

    return {"documents": relevant_documents, "relevance_score": average_score}


def rewrite_query(state: RAGState):

    query = state["query"]
    retry_count = state["retry_count"]

    print(f"Attempt {retry_count}. Improving query")

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a query rewriter for a RAG system.
The original query didn't retrieve relevant documents.

Rewrite the query to be more specific and likely to match relevant documents.
Consider:
- Adding synonyms or related terms
- Being more specific about what information is needed
- Rephrasing to match how documentation is typically written

Output ONLY the rewritten query, nothing else.""",
            ),
            (
                "human",
                """Here is the initial query:
{query}

Formulate an improved query:""",
            ),
        ]
    )

    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    rewritten = result.content

    print(f"Original query ->>> {query}")
    print(f"Rewritten query ->>> {rewritten}")

    return {"rewritten_query": rewritten, "retry_count": retry_count + 1}


def generate_answer(state: RAGState):

    query = state["query"]
    documents = state["documents"]

    print(f"Generating answer from {len(documents)}")

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get("source", "unknown")}\n{doc.page_content}"
            for doc in documents
        ]
    )

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant answering questions based on provided context.

Use ONLY the information in the context to answer. If the context doesn't contain enough information, say so clearly.

Always cite your sources by mentioning which document the information came from.""",
            ),
            (
                "human",
                """Context:
{context}

Answer the following question:
{query}""",
            ),
        ]
    )

    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "query": query})
    print("Answer generated")

    return {"generation": result.content}


def generate_fallback(state: RAGState):

    query = state["query"]
    print(f"Retrieval failed after {state.get("retry_count"), 0} attempts")

    fallback_message = f"""I couldn't find relevant information to answer your question: '{query}'.

This could mean:
1. The information isn't in my knowledge base
2. Try rephrasing your question with different terms
3. The topic might not be covered in the available documents

Would you like to try a different question?"""

    return {"generation": fallback_message}


def should_retry_or_generate(state: RAGState):

    relevance_score = state.get("relevance_score", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)
    documents = state.get("documents", [])

    print(f"Evaluating: score={relevance_score}, retries={retry_count}/{max_retries}, ")

    if relevance_score >= 0.5 and len(documents) > 0:
        print("GENERATE(good relevance)")
        return "generate"

    if retry_count < max_retries:
        print("Low relevance -> retrying")
        return "rewrite"

    if len(documents) > 0:
        print("GENERATE(out of retries, using available docs)")
        return "generate"
    else:
        print("Fallback -> (no relevant documents)")
        return "fallback"


def build_agentic_rag_graph():

    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", generate_fallback)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_retry_or_generate,
        {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"}
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    return app

def run_demo():

    print("Setting up vectorstore")
    vectorstore = create_vectorstore()

    print("Building agentic rag graph")
    app = build_agentic_rag_graph()

    test_queries = [
        "How can I open new visa card?",
        "Are any documents required for opening a new plastic card?",
        "How can I run fast?"
    ]

    for query in test_queries:

        initial_state = {
            "query": query,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "retry_count": 0,
            "max_retries": 2,
            "_vectorstore": vectorstore
        }  

        result = app.invoke(initial_state)
        print("Final answer")
        print(result['generation'])

    vectorstore.delete_collection()

run_demo()