import os
from flask import Flask, request
from flask_cors import CORS, cross_origin
from langchain_ai21 import ChatAI21
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ai21 import AI21Embeddings
from pytube import Playlist
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.agents import AgentExecutor, create_react_agent
import os
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer

app = Flask(__name__)
os.environ["AI21_API_KEY"] = "tQ5ybFtCPDfMCgXZiCdVLwejMyKYro8p"

api_v1_cors_config = {
    "origins": ["*"], 
    "methods": ["OPTIONS", 'GET', 'POST'], 
    "allow_headers": ["Authorization", "Content-Type"]
}

PLAYLISTID_TO_RETRIEVER = {}

PLAYLISTID_TO_DOCUMENTS = {}

PLAYLISTID_TO_ONE_LINERS = {}

CORS(app, resources={r"/*" : api_v1_cors_config})

@app.get('/')
@cross_origin(**api_v1_cors_config)
def hello_world():
    return 'Hello, World!'

@app.post('/setup')
@cross_origin(**api_v1_cors_config)
def setup():
    body = request.get_json()
    playlist_URL = body['playlist_URL']

    PLAYLIST_ID = playlist_URL[playlist_URL.rfind("=") + 1 : ]

    # if PLAYLIST_ID in PLAYLISTID_TO_RETRIEVER: 
    #      return {'hello': "ooga"}
    
    playlist_urls = Playlist(playlist_URL)
    
    if not os.path.isdir(f"videos/{PLAYLIST_ID}"): 
        os.mkdir(f"videos/{PLAYLIST_ID}")

    for url in playlist_urls: 
        video_id = url[url.rfind("=") + 1 : ]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        string = " ".join([transcript_time['text'] for transcript_time in transcript])

        if os.path.isfile(f"videos/{PLAYLIST_ID}/{video_id}.txt"): continue 

        f = open(f"videos/{PLAYLIST_ID}/{video_id}.txt", "w")
        f.write(string.replace('\n', ""))
        f.close()
    
    loader = DirectoryLoader(
        path = f"./videos/{PLAYLIST_ID}",
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
    
    PLAYLISTID_TO_RETRIEVER[PLAYLIST_ID] = retriever
    PLAYLISTID_TO_DOCUMENTS[PLAYLIST_ID] = documents

    print(PLAYLISTID_TO_RETRIEVER)
    
    return {'hello': "ooga"}

@app.post('/summary')
@cross_origin(**api_v1_cors_config)
def summary():
    body = request.get_json()
    PLAYLIST_ID = body['playlist_id']
    
    documents = PLAYLISTID_TO_DOCUMENTS[PLAYLIST_ID]
    concat_docs = ""
    for doc in documents:
        concat_docs += doc.dict()['page_content']

    template = """
    Given this context: {context}

    Please do: {question}
    """

    prompt = PromptTemplate.from_template(template)

    # model = ChatAI21(model="jamba-instruct", max_tokens=500)

    # model = ChatHuggingFace(model_id="mistralai/Mistral-7B-Instruct-v0.3", max_tokens=500)

    # https://python.langchain.com/v0.2/docs/integrations/chat/huggingface/

    model = ChatHuggingFace(llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=4096, 
        top_k=10,
        top_p=0.95
        # max_new_tokens=4096,
        # do_sample=False,
    ))

    chain = prompt | model

    response = chain.invoke({"context":concat_docs, "question": "Please give a detailed summary of the playlist 2-3 sentences."}).dict()["content"]
    one_line_response = chain.invoke({"context":concat_docs, "question": "Please give a brief, one sentence summary."}).dict()["content"]
    PLAYLISTID_TO_ONE_LINERS[PLAYLIST_ID] = one_line_response
    print(f"this is the response: {response}")
    return {"summary" : response}


@app.post('/qa')
@cross_origin(**api_v1_cors_config)
def qa(): 
    body = request.get_json()
    PLAYLIST_ID = body['playlist_id']
    QUESTION = body['question']
    chat_llm = ChatAI21(model="jamba-instruct", max_tokens=512, temperature=0)
    retriever = PLAYLISTID_TO_RETRIEVER[PLAYLIST_ID]

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

    agent = create_react_agent(chat_llm, tools, agent_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    out = agent_executor.invoke({"input": QUESTION})
    output = out['output']
    answer = None 
    if '\nObservation' in output: 
        answer = output[:output.index('\nObservation')]
    else: 
        answer = output 

    return {"answer": answer}