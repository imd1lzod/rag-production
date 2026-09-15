import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import networkx as nx

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


def build_knowledge_graph():

    G = nx.DiGraph()

    entities = [
        ("John Smith", {"type": "Person", "role": "CEO"}),
        ("Sarah Johnson", {"type": "Person", "role": "Executive Assistant"}),
        ("Mike Brown", {"type": "Person", "role": "CFO"}),
        ("Lisa Chen", {"type": "Person", "role": "CLO"}),
        ("TechCorp", {"type": "Organization"}),
        ("Executive Department", {"type": "Department", "floor": "5th"}),
    ]

    for entity, attrs in entities:
        G.add_node(entity, **attrs)

    relationships = [
        ("John Smith", "TechCorp", {"relation": "CEO_OF"}),
        ("Sarah Johnson", "John Smith", {"relation": "ASSISTANT_TO"}),
        ("Sarah Johnson", "Executive Department", {"relation": "WORKS_IN"}),
        ("Mike Brown", "Executive Department", {"relation": "WORKS_IN"}),
        ("Lisa Chen", "Executive Department", {"relation": "WORKS_IN"}),
        ("John Smith", "Executive Department", {"relation": "WORKS_IN"}),
    ]

    for source, target, attrs in relationships:
        G.add_edge(source, target, **attrs)

    return G


def traverse_graph_for_answer(G: nx.DiGraph):

    query = "Who works in a same department as the CEO's assistant?"

    print("Step 1: Find the CEO")
    ceo = None
    for node, attrs in G.nodes(data=True):
        if attrs.get("role") == "CEO":
            ceo = node
            break

    print("\nStep 2: Find the CEO's assistant")
    assistant = None
    for source, target, attrs in G.edges(data=True):
        if target == ceo and attrs["relation"] == "ASSISTANT_TO":
            assistant = source
            print(f" Found: {assistant} --[ASSISTANT_TO]--> {ceo}")
            break

    print("\nStep 3: Find the assistant's department")
    department = None
    for source, target, attrs in G.edges(data=True):
        if source == assistant and attrs["relation"] == "WORKS_IN":
            department = target
            print(f" Found: {assistant} --[WORKS_IN]--> {department}")
            break

    print("\nStep 4: Find others in the same department")
    coworkers = []
    for source, target, attrs in G.edges(data=True):
        if target == department and attrs["relation"] == "WORKS_IN":
            if source != assistant:  # Exclude the assistant themselves
                coworkers.append(source)
                print(f" Found: {source} --[WORKS_IN]--> {department}")

    print(f"The CEO's assistant is {assistant}.")
    print(f"{assistant} works in the {department}.")
    print(f"Others in the same department: {', '.join(coworkers)}")


def extract_entities_with_llm(text: str):

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at extracting knowledge graph elements from text.

Given a text, extract:
1. ENTITIES: People, organizations, places, concepts
2. RELATIONSHIPS: How entities are connected

Output in this exact format:
ENTITIES:
- [TYPE] Name: description

RELATIONSHIPS:
- Source --[RELATION]--> Target

Be specific about relationship types (WORKS_FOR, MANAGES, LOCATED_IN, etc.)""",
            ),
            (
                "human",
                """Extract entities and relationships from this text:

{text}""",
            ),
        ]
    )

    sample_text = """
Acme Corporation announced today that Jennifer Lee has been appointed as the new 
Chief Technology Officer. She will report directly to CEO Marcus Chen. Jennifer 
previously led the AI research team at DataTech Inc. in Boston. The company's 
headquarters will remain in San Francisco, where the engineering team of 50 
developers is based.
"""

    chain = extraction_prompt | llm
    result = chain.invoke({"text": sample_text})

    return result.content

if __name__ == "__main__":
    graph = build_knowledge_graph()

    traverse_graph_for_answer(graph)

    ll_output = extract_entities_with_llm("")
    print(ll_output)