import uuid
from ai.agent import AthenaAgent
from datetime import datetime

def test_memory_and_rag():
    """Tests the memory and RAG functionalities of the AthenaAgent."""

    user_id = uuid.uuid4()
    agent = AthenaAgent(user_id)

    print("Adding knowledge to the knowledge base...")
    agent.add_to_knowledge_base(
        content="The capital of France is Paris.",
        metadata={"source": "test_data", "topic": "geography"}
    )
    agent.add_to_knowledge_base(
        content="Python is a high-level programming language.",
        metadata={"source": "test_data", "topic": "programming"}
    )

    print("\nRetrieving relevant memories for 'capital of France'...")
    response = agent.process_input("What is the capital of France?")
    print(f"Agent Response: {response}")

    print("\nSimulating a short conversation...")
    agent.process_input("How are you?")
    agent.process_input("What's the weather like?")

    print("\nRetrieving relevant memories for 'weather'...")
    response = agent.process_input("What's the weather like?")
    print(f"Agent Response: {response}")

    print("\nAdding a document to RAG...")
    agent.rag_system.add_document(
        content="""
        Quantum computing is a rapidly-emerging technology that harnesses the laws of 
        quantum mechanics to solve problems too complex for classical computers. 
        It uses quantum bits or qubits, which can represent 0, 1, or a combination 
        of both states simultaneously, unlike classical bits that can only be 0 or 1.
        This property, known as superposition, along with other quantum phenomena 
        like entanglement and interference, allows quantum computers to perform 
        certain types of calculations much faster than classical computers.
        """,
        metadata={"source": "test_doc", "topic": "quantum computing"},
        doc_id="quantum_computing_doc"
    )

    print("\nRetrieving using RAG for 'quantum computing'...")
    response = agent.process_input("Explain quantum computing")
    print(f"Agent Response: {response}")

    print("\nMemory and RAG System Metrics:")
    print(agent.get_metrics())

if __name__ == "__main__":
    test_memory_and_rag()
