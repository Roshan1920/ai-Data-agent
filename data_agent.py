from dotenv import load_dotenv
load_dotenv()

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import os

# 🔑 Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# 📊 Load your dataset
df = pd.read_csv("data.csv")

# ⚙️ Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini")

# 🤖 Create the agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

# 💬 Ask questions
while True:
    query = input("\nAsk a question about your data (or type 'exit'): ")
    if query.lower() == "exit":
        break
    response = agent.invoke(query)
    print("\nAI:", response["output"])
