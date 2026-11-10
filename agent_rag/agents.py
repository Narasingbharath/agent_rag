import os
from crewai import Agent, LLM

os.environ["AWS_ACCESS_KEY_ID"] = "AKIA6QKHEVBYCDYZQNN3"
os.environ["AWS_SECRET_ACCESS_KEY"] = "u1wBrSzq911FoRsCn+cE2DybdZ7TySz3ei71OBZN"
# If using temporary credentials:
# os.environ["AWS_SESSION_TOKEN"] = "your_session_token"
os.environ["AWS_REGION_NAME"] = "us-east-1"


# Choose any Bedrock model you've enabled in the Bedrock console
llm = LLM(
    model="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0",
    temperature=0.2,
    max_tokens= 4000,
)


# PDF QA Agent
rag_agent = Agent(
    role="PDF Q&A Assistant",
    goal=(
        "Answer user questions based only on the PDF content. "
        "If the answer is not in the PDF, reply with 'I don’t know the answer.'"
    ),
    backstory=(
        "You are a helpful assistant trained to answer strictly "
        "from the uploaded PDF document."
    ),
    llm=llm
)
