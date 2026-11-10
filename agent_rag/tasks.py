from crewai import Task
from agents import rag_agent
rag_task = Task(
    description=(
        "Answer the user's question based strictly on the provided PDF context. "
        "If the answer is not available in the context, say 'I don’t know the answer.'"
    ),
    expected_output="A direct and accurate answer, or 'I don’t know the answer.'",
    agent = rag_agent
)
