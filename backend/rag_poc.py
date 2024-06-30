import os
import langchain
from langchain import hub
from langchain_ai21 import AI21LLM
from langchain_ai21 import ChatAI21
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ai21 import AI21Embeddings, AI21ContextualAnswers
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import Document


os.environ["AI21_API_KEY"] = "tQ5ybFtCPDfMCgXZiCdVLwejMyKYro8p"
os.environ["TAVILY_API_KEY"] = "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"

loader = DirectoryLoader(
    path = "./videos",
    glob="**/*.txt",
    show_progress = True
)

documents = loader.load()
# insert splitting/chunking step here
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # going to have to improve this most likely 
splits = text_splitter.split_documents(documents=documents)
embeddings = AI21Embeddings()
vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"top_k": 5})

out = retriever.invoke("What is a Lewis structure?")

print(f"Retrieved Document Example: {out[0]}")


decision_llm = ChatAI21(model="jamba-instruct", max_tokens=1024, temperature=0)

chat_llm = ChatAI21(model="jamba-instruct", max_tokens=1024)

tool = create_retriever_tool(
    retriever,
    "search_video_transcripts",
    "searches and returns relevant parts of the video transcripts",
)
tools = [tool]
agent_prompt = hub.pull("hwchase17/react")

agent_prompt.template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

#agent = create_react_agent(chat_llm, tools, agent_prompt)

#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

#out = agent_executor.invoke({"input": "what are Lewis Structures?"})

#print(out)











from typing import List
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph, MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults


class GraphState(TypedDict):
    question: str
    query: str
    documents: List[str]
    generation: str
    num_loops: int

# define chains --- CHANGE THIS PROMPT TO HAVE SUMMARY OF PLAYLIST.
question_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. \n
    Use the vectorstore for questions on chemistry-related topics. \n
    You do not need to be stringent with the keywords in the question related to these topics. \n
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
    Return either the string 'web_search' or 'vectorstore' as output, without any preamble or explanation.\n
    Question to route: {question}""",
    input_variables=["question", "summary"],
)

question_router = question_router_prompt | decision_llm | StrOutputParser()

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as either 'yes' or 'no', without any preamble or explanation. """,
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | decision_llm | StrOutputParser()

rag_prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = rag_prompt | chat_llm | StrOutputParser()

grounding_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as either 'yes' or 'no', without any preamble or explanation.""",
    input_variables=["generation", "documents"],
)

grounding_grader = grounding_grader_prompt | decision_llm | StrOutputParser()

gen_vector_query_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input query into a better version that is optimized \n for vectorstore retrieval. Look at the initial question, extract key terms, remove amiguities, and prioritize the most specific and relevant terms.\n
     Here is the initial question: \n\n {question}. Provide the improved query as just a string, with no preamble or explanation.\n """,
    input_variables=["question"],
)

gen_vector_query_tool = gen_vector_query_prompt | decision_llm | StrOutputParser()

gen_web_query_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input query into a better version that is optimized \n for websearch retrieval. Look at the initial question, extract key terms, remove amiguities, and prioritize the most specific and relevant terms.\n
     Here is the initial question: \n\n {question}. Provide the improved query as just a string, with no preamble or explanation.\n """,
    input_variables=["question"],
)

gen_web_query_tool = gen_web_query_prompt | decision_llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

# define nodes
def gen_vector_query(state):
    print("Generating vectorstore query...")
    question = state["question"]
    query = gen_vector_query_tool.invoke({"question": question})
    return {"query": query, "num_loops": 0}

def gen_web_query(state):
    print("Generating websearch query...")
    question = state["question"]
    query = gen_web_query_tool.invoke({"question": question})
    return {"query": query, "num_loops": 0}

def retrieve(state):
    print("Retrieving...")
    query = state["query"]
    documents = retriever.invoke(query)
    return {"documents": documents}

def generate(state):
    print("Generating...")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def web_search(state):
    print("Searching web...")
    query = state["query"]
    documents = web_search_tool.invoke({"query": query})
    web_results = [Document(page_content=d["content"]) for d in documents]
    return {"documents": web_results}

def filter_docs(state):
    print("Filtering Docs...")
    question = state["question"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
    return {"documents": filtered_docs, "question": question}

def generate_final(state):
    print("Generating Final Output...")
    question = state["question"]
    generation = chat_llm.invoke(question)
    return {"documents": documents, "question": question, "generation": generation}

# define edges
def route_question(state):
    print("Routing question...")
    question = state["question"]
    print(question)
    result = question_router.invoke({"question": question})
    if result == "web_search":
        return "web_search"
    elif result == "vectorstore":
        return "vectorstore"

def decide_to_generate(state):
    filtered_docs = state["documents"]
    num_loops = state["num_loops"]
    if not filtered_docs:
        #print(str(num_loops) + "---------------")
        state["num_loops"] = num_loops + 1
        #print(state["num_loops"])
        return "web_search"
    elif num_loops > 5:
        print("Sending straight to generation.")
        return "generate_final"
    else:
        return "generate"

def grade_generation(state):

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    num_loops = state["num_loops"]
    score = grounding_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score

    # Check hallucination
    if grade == "yes":
        print("Output grounded, good to go!")
        return "grounded"
    elif num_loops > 5:
        print("Sending straight to generation.")
        return "generate_final"
    else:
        print("Output not grounded, sending back to generate.")
        #print(str(num_loops) + "---------------")
        state["num_loops"] = num_loops + 1
        #print(state["num_loops"])
        return "not grounded"

# build graph

graph = StateGraph(GraphState)

graph.add_node("gen_vec_query", gen_vector_query)
graph.add_node("gen_web_query", gen_web_query)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("web_search", web_search)
graph.add_node("filter_docs", filter_docs)
graph.add_node("generate_final", generate_final)

graph.set_conditional_entry_point(
    route_question,
    {
        "web_search": "gen_web_query",
        "vectorstore": "gen_vec_query",
    }
)

graph.add_edge("gen_vec_query", "retrieve")
graph.add_edge("retrieve", "filter_docs")
graph.add_edge("gen_web_query", "web_search")
graph.add_edge("web_search", "filter_docs")

graph.add_conditional_edges(
    "filter_docs",
    decide_to_generate,
    {
        "web_search": "gen_web_query",
        "generate": "generate",
        "generate_final": "generate_final"
    }
)

graph.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "grounded": END,
        "not grounded": "generate",
        "generate_final": "generate_final"
    }
)

app = graph.compile()

from pprint import pprint

inputs = {"question": "Explain to me how to determine if an ozone molecule is polar."}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])

"""
print(documents)
concat_docs = ""
for doc in documents:
    concat_docs += doc.dict()['page_content']

template = 
Given this context: {context}

Please do: {action}


prompt = PromptTemplate.from_template(template)

model = AI21LLM(model="jamba-instruct")

chain = prompt | model

chain.invoke({"context":concat_docs, "question": "Please give a brief summary of this context."})


"""


# """
# tsm = AI21ContextualAnswers()
# chain = tsm | StrOutputParser()

# response = chain.invoke(
#     {"context": "Your context", "question": "What is a Lewis structure"},
# )

# print(response)
