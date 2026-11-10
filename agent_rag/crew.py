from crewai import Crew
from agents import rag_agent
from tasks import rag_task

rag_crew = Crew(
    agents=[rag_agent],
    tasks=[rag_task],
    verbose=True
)
