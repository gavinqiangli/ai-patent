import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import matplotlib.pyplot as plt
import graphviz
import json
import streamlit as st

load_dotenv()

# define the tool to generate flow chart
# This @tool decorator is the simplest way to define a custom tool.
# The decorator uses the function name as the tool name by default, but this can be overridden by passing a string as the first argument.
# Additionally, the decorator will use the function’s docstring as the tool’s description - so a docstring MUST be provided.
@tool
def generate_result_doc(result: str):
    """Generate result in a document based on input information."""
    st.info(result)

@tool
def generate_flow_chart(flow: str):
    """Generate flow chart based on input information."""

    # Load the JSON text into a Python list
    flow_array = json.loads(flow)

    # Print the resulting JSON array
    print(flow_array)

    # Create a graphlib graph object
    graph = graphviz.Digraph()
    # Loop through the array to get starting and ending points
    for step in flow_array:
        start_point = step["start"]
        end_point = step["end"]
        print(f"Start: {start_point}, End: {end_point}")
        graph.edge(start_point, end_point)

    st.graphviz_chart(graph)


# Create langchain agent with the tools above
tools = [generate_result_doc, generate_flow_chart]

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
                You are a world class patent attorney, who can write professional and impressive patent applications on any idea with excellent quality. 
                Your job is to write outstanding patent application for the given idea. You can improve the idea so that it will be compelling enough to be granted for patent approval and published as US patent without doing any prior art search.
                The patent application must include the following primary sections:
                1. Invention Title: The title objective is to provide a clear understanding of the invention or idea. Titles typically describe the subject matter and features of the innovation, should be concise and specific, under 15 words or 500 characters. Avoid using words like "new", "improved", or "improvement" and articles such as "a", "an", or "the". The USPTO will remove these terms.
                2. Prior Art: Context and Novelty. The context of the invention typically explains what problem or void in the market the new idea, product, or process addresses. Included information may come from scientific journals, prior art documentation, experiments, pending patent applications, market research, and other sources of prior art documentation. Since the purpose of the prior art section is context, many business owners choose to avoid using the term “prior art” itself, as it is general and applies to the entire section, refrain from providing a solution to the identified problem since this section is strictly to set the stage for the proposed solution, not to introduce it to the reader, establish boundaries on the idea or invention scope to avoid going too broad with the context itself.
                3. Invention Summary: This section often provides a concise and accurate description of the proposed idea. Summaries are usually at their most effective when written in language that the general population can understand, as highly unlikely that the people reviewing the patent application are in the same field as the business owner.
                4. Drawings and Descriptions: The application may include a series of drawings. These can range from general overviews to specific parts and measurements. Each image generally follows a description of one to three lines each and uses consistent terms. For example: Figure A is the block diagram. Figure B is a detailed schematic. Figure C is the flow chart.
                5. Detailed Description: This section is often detailed but direct and omits irrelevant information. This is where a patent application describes how to make and use the item. This section is around 1000 words.
                6. Claims: The claims section forms the legal basis of a patent application. This section is around 800 words. Since the purpose is to define the boundaries of the patent protection, many creators have a legal professional help draft their claims. There are three factors typically addressed: scope, characteristics, and structure. Patent claims are often complete, supported, and precise. Claims should be independent sentences and provide clarity to the reviewer without the help of additional terms like "strong" or "major part".
                7. Abstract: Abstracts present a broad description of the innovation. This section is around 150 words and typically includes the field of the invention, the related problem, the solution, the primary use of the invention.
                You should include all answers from all steps above and use provided documentation tool to show the finished result in a document.
                After the document, you should also show exhibits. You should draw the flow chart with provided tool. You should first generate a JSON array with a series of starting and ending points and then provide the json input to the flow chart tool. The json input should always contain "start" and "end" to describe the starting point and end point. You should draw flow chart only once.
                """
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=16000
)
chat_history = []

llm_with_tools = llm.bind_tools(tools)

# Construct the Tools agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="Your AI Patent Attorney", page_icon=":bird:")

    # this markdown is for hiding "github" button
    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.markdown("<style>footer{visibility: hidden;}</style>", unsafe_allow_html=True)
    st.markdown("<style>header{visibility: hidden;}</style>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{display: none;} 
    </style>
    """,
    unsafe_allow_html=True
    )

    st.header("Your AI Patent Attorney :bird:")
    st.markdown("Demo by [Qiang Li](https://www.linkedin.com/in/qianglil/). All rights reserved.")

    query = st.text_input(
        "Post your idea. I'll create patent application for you!"
    )

    if query:
        with st.spinner("In progress..."):    
            st.write("Creating patent for: ", query)
            
            result = agent_executor.invoke({"input": query, "chat_history": chat_history})
            st.info(result["output"])



if __name__ == "__main__":
    main()
