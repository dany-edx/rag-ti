import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os 
import re
import requests
from bs4 import BeautifulSoup
import io
from io import StringIO, BytesIO
from tqdm.auto import tqdm
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
import llama_index.core
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import get_response_synthesizer,  ServiceContext, Document
from llama_index.core.schema import Document
import nest_asyncio
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append('/home/qcells/Desktop/rag_project/utils')
sys.path.append("/workspaces/hanwhaqcells/utils")
sys.path.append('../utils')
from qcells_web_instance_search import instance_search_expanding
from qcells_route_engine import qcell_engine, web_engine, Decide_to_search_external_web, GoogleRandomSearchToolSpec, VectordbSearchToolSpec
from qcells_custom_rag import create_db_chat, pptx_load_data, pdf_load_data, get_youtube_metadata, get_timeline
from web_crack import RemoteDepthReader
from youtube_transcript_api import YouTubeTranscriptApi
from streamlit_pdf_viewer import pdf_viewer
import phoenix as px
import base64
import os
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
nest_asyncio.apply()
st.set_page_config(page_title="RAG",  layout="wide",  page_icon="☀️")

class global_obj(object):
    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("detach", True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}

@st.cache_resource
def lanch_px_app():
    llama_index.core.set_global_handler("arize_phoenix")
    session = px.launch_app(host='0.0.0.0')
    
lanch_px_app()
def set_llm():
    return AzureOpenAI(
            model="gpt-35-turbo",
            deployment_name="qcell_gpt_model",
            temperature = 0,
            api_key="c11ed4df2d35412b89a7b51a631bf0e4",
            azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
            api_version="2024-02-15-preview")
def set_rag():
    return AzureOpenAI(
            model="gpt-35-turbo",
            deployment_name="qcell-gpt-model-rag",
            temperature = 0,
            api_key="c11ed4df2d35412b89a7b51a631bf0e4",
            azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
            api_version="2024-02-15-preview")
def set_llm4():
    return AzureOpenAI(
            model="gpt-4",
            deployment_name="qcell_gpt4_model",
            temperature = 0,
            api_key="2b6d6cbbc0ae4276aad07db896f63bfd",
            azure_endpoint="https://rag-openai-qcells-norway.openai.azure.com/",
            api_version="2024-02-15-preview")
def set_embedding():
    return AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="qcell_embedding_model",
        api_key="c11ed4df2d35412b89a7b51a631bf0e4",
        azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
     api_version="2023-07-01-preview")
def get_answer_yn(llm, query_str, text_chunks):
    synthesizer = get_response_synthesizer(llm = llm, response_mode="refine", output_cls = Decide_to_search_external_web)
    result_response = synthesizer.get_response(query_str = query_str, text_chunks=[text_chunks], verbose = True)      
    return result_response
if "llm" not in st.session_state:
    st.session_state.llm = set_llm()
if "llm_rag" not in st.session_state:
    st.session_state.llm_rag = set_rag()
if "embedding" not in st.session_state:
    st.session_state.embedding = set_embedding()
if "llm4" not in st.session_state:
    st.session_state.llm4 = set_llm4()
if "rag" not in st.session_state:    
    st.session_state.rag = qcell_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
if "webrag" not in st.session_state:
    st.session_state.webrag = web_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
if "service_context" not in st.session_state:
    st.session_state.service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)
if "doc_read_yn" not in st.session_state:
    st.session_state.doc_read_yn = False
if "youtube_read_yn" not in st.session_state:
    st.session_state.youtube_read_yn = False
if "webpage_read_yn" not in st.session_state:
    st.session_state.webpage_read_yn = False
if "webpageall_read_yn" not in st.session_state:
    st.session_state.webpageall_read_yn = False
if "external_docs" not in st.session_state:
    st.session_state.external_docs = []
if "display_datasource" not in st.session_state:
    st.session_state.display_datasource = []
if "chat_db" not in st.session_state:
    st.session_state.chat_db = None
if "messages1" not in st.session_state.keys():
    st.session_state.messages1 = [{"role": "system", "content": "Hello, What can I do for you?"}]
if "messages2" not in st.session_state.keys(): 
    st.session_state.messages2 = [{"role": "system", "content": "Hello, What can I do for you?"}]
if "messages3" not in st.session_state.keys(): 
    st.session_state.messages3 = [{"role": "system", "content": "Hello, What can I do for you?"}]
if "messages4" not in st.session_state.keys(): 
    st.session_state.messages4 = [{"role": "system", "content": "Hello, What can I do for you?"}]
if "chosen_id" not in st.session_state:
    st.session_state.chosen_id = 'ChatGPT 3.5'
if "chat_engine2" not in st.session_state.keys(): 
    st.session_state.chat_engine2 = None
if "multiple_files" not in st.session_state.keys(): 
    st.session_state.multiple_files = ''
if "prompts1" not in st.session_state.keys(): 
    st.session_state.prompts1 = []
if "prompts2" not in st.session_state.keys():     
    st.session_state.prompts2 = []
if "prompts3" not in st.session_state.keys():     
    st.session_state.prompts3 = []
if "prompts4" not in st.session_state.keys(): 
    st.session_state.prompts4 = []
if "youtube_embeded_html" not in st.session_state.keys(): 
    st.session_state.youtube_embeded_html = []
if "img_embeded_html" not in st.session_state.keys(): 
    st.session_state.img_embeded_html = []

def make_data_instance():
    document_uploader()
    youtube_uploader()
    web_uploader()
    with st.spinner("Creating knowledge database"):
        st.session_state.chat_db = create_db_chat(st.session_state.external_docs, st.session_state.llm_rag, st.session_state.embedding,  st.session_state.service_context)
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
    st.session_state.chat_engine2 = ReActAgent.from_llm(st.session_state.chat_db.to_tool_list(), memory = memory, llm = st.session_state.llm_rag, verbose = True)    
    st.session_state.external_docs = []
    
def chat_box(text):
    texts = text.split('```')
    for i in texts:
        if i.split('\n')[0] ==  'python':
            with stylable_container(
                "codeblock2",
                """
                div {background-color:rgb(0,0,0,0.1); border-radius:10px} 
                
                code {font-size:13px;font-family: Arial;}
                """,):
                x = st.code(i, language = 'python', line_numbers = True)
        else:
            with stylable_container(
                "codeblock",
                """
                div {background-color:rgb(0,0,0,0)}
                code {font-size:13px; font-family: Arial; white-space: pre-wrap !important;}
                """,):
                x = st.code(i, language = 'md')

def message_hist_display(message_history):
    for idx, message in enumerate(message_history):
        if message["role"] == 'assistant':
            avatar = './src/chatbot.png'
        elif message["role"] == 'system':
            avatar = './src/chatbot.png'
        else:
            avatar = './src/human.png'                
        with st.chat_message(message["role"], avatar = avatar):
            msg = message["content"]
            # if (idx == 0) or (message["role"] != 'system'):
            chat_box(msg)

def websearch_func(prompt, response):
    answer_eval = get_answer_yn(st.session_state.llm, prompt, response)
    print(answer_eval)
    res = None
    if answer_eval.Succeed_answer == False:     
        if answer_eval.Decide_web_search == True:
            if answer_eval.Reason == True:
                with st.spinner("Google Search..."):
                    ise = ''
                    try:
                        ise = instance_search_expanding(query = answer_eval.Searchable_query)
                        st.session_state.youtube_embeded_html = ise.embed_html
                        st.session_state.img_embeded_html = ise.img_list
                        res = st.session_state.webrag.chat(answer_eval.Searchable_query)
                        res = res.response
                    except Exception as e:
                        res = 'retry! ' + str(e)
    return res
    
# @st.cache_resource(experimental_allow_widgets=True)
def display_customized_data(_source):
    for idx, i in enumerate(_source):
        if 'pdf' in list(i.keys())[0]:
            pdf_viewer(input=i['pdf'], width=550, key = 'tmp' + str(idx))
        if 'youtube' in list(i.keys())[0]:
            st.write('<iframe src="{}" width="100%" height="400px"></iframe>'.format(i['youtube']),unsafe_allow_html=True,)
        if 'HTML' in list(i.keys())[0]:
            st.write(i['HTML'])

def convert_message(x):
    return [ChatMessage(role=i['role'], content=i['content']) for i in x]        

def reset_conversation(x):
    st.session_state.llm = set_llm()
    st.session_state.llm4 = set_llm4()
    st.session_state.embedding = set_embedding()
    st.session_state.service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)
    if x == 'ChatGPT 3.5':
        st.session_state.chat_engine = st.session_state.llm
        st.session_state.webrag.reset()
        st.session_state.webrag = web_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
        st.session_state.messages1 = [{"role": "system", "content": "Hello, What can I do for you?"}]    
        st.session_state.prompts1 = []
    if x == 'ChatGPT 4':
        st.session_state.chat_engine_llm4 = st.session_state.llm4
        st.session_state.webrag.reset()
        st.session_state.webrag = web_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
        st.session_state.messages2 = [{"role": "system", "content": "Hello, What can I do for you?"}]    
        st.session_state.prompts2 = []
    if x == 'ChatGPT+TechSensing':
        st.session_state.rag.reset()
        st.session_state.rag = qcell_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
        st.session_state.messages3 = [{"role": "system", "content": "Hello, What can I do for you?"}]            
        st.session_state.prompts3 = []
    if x == 'ChatGPT+MyData':
        st.session_state.messages4 = [{"role": "system", "content": "Hello, What can I do for you?"}]    
        st.session_state.prompts4 = []
    st.cache_resource.clear()
    st.session_state.display_datasource = []
    
def document_uploader():        
    if len(st.session_state.multiple_files) > 0:
        st.session_state.external_docs = []
        for doc in st.session_state.multiple_files:
            if doc.name.split('.')[-1] == 'pdf':
                string_data = pdf_load_data(doc)
                pdf_documents = [Document(text=string_data, metadata= {"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(pdf_documents)  
                st.session_state.display_datasource.append({'pdf': doc.getvalue()})
            if doc.name.split('.')[-1] == 'py':
                stringio = StringIO(doc.getvalue().decode("utf-8"))
                string_data = stringio.read()
                document = [Document(text=string_data, metadata= {"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(document)  
            if doc.name.split('.')[-1] == 'pptx':
                result = pptx_load_data(doc) 
                document = [Document(text=result, metadata={"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(document)    

def youtube_uploader():        
    if len(st.session_state.youtube_data) > 0:
        meta_data = get_youtube_metadata(st.session_state.youtube_data)
        st.session_state.display_datasource.append({'youtube': meta_data['embed_url']})
        data = YouTubeTranscriptApi.get_transcript(st.session_state.youtube_data.split('v=')[-1])
        documents = []
        for i in data:
            i['div'] = int(i['start'] / 60)  
        total_text= []
        distinct_div = set(item['div'] for item in data)
        for idx, d in enumerate(distinct_div):
            texts = []
            k = [i for i in data if i['div'] == d]
            start_time_min = min([i['start'] for i in k])
            start_time_max = max([i['start'] for i in k])
            text_div = [i['text'] for i in k]
            texts.append('[youtube play time] {}\n'.format(get_timeline(start_time_min)) + ' '.join(text_div))
            texts = '\n'.join(texts)
            total_text.append(texts)            
        total_text = '\n'.join(total_text)
        documents.append(Document(text=total_text, metadata= {'title':  meta_data['title'], 'resource' : 'youtube'}))
        st.session_state.external_docs.append(documents)

def web_uploader(): 
    if len(st.session_state.single_page_data) > 0:
        driver = webdriver.Chrome( options=global_obj.chrome_options)
        driver.delete_all_cookies()
        driver.get(st.session_state.single_page_data) 
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        anchor_elements = soup.find_all('p')
        result_string = '\n'.join([i.text for i in anchor_elements])                
        web_data_documents = [Document(text=result_string, metadata= {'title':  st.session_state.single_page_data, 'resource' : 'web_page'})]
        st.session_state.display_datasource.append({'HTML': result_string})
        st.session_state.external_docs.append(web_data_documents)
        
st.markdown('''
            <style>
                body {font-size: 7px}
                img {border-radius: 10px;}
                div[data-testid="stExpander"]> details {border-width: 0px;} 
                div[data-testid="stCodeBlock"]> pre {background: rgb(248, 249, 255, 0);} 
            </style>
            ''',unsafe_allow_html=True,)

st.sidebar.write('''<img width="180" height="60" src="https://us.qcells.com/wp-content/uploads/2023/06/qcells-logo.svg"  alt="Qcells"><br><br>''',unsafe_allow_html=True,)
col1, col2  = st.columns([1, 3])
with col1:
    st.session_state.chosen_id = st.selectbox('', ('ChatGPT 3.5', 'ChatGPT 4', 'ChatGPT+TechSensing', 'ChatGPT+MyData'), label_visibility="collapsed", key = 'model_select')
with col2:
    st.button('Reset Chat', on_click=reset_conversation,args=[st.session_state.chosen_id], key = 'reset1')


if prompt := st.chat_input("Your question", key = 'chat_input_query'): # Prompt for user input and save to chat history
    if st.session_state.chosen_id == 'ChatGPT 3.5':
        st.session_state.prompts1.append(prompt)
        st.session_state.messages1.append({"role": "user", "content": prompt})
    if st.session_state.chosen_id == 'ChatGPT 4':
        st.session_state.prompts2.append(prompt)
        st.session_state.messages2.append({"role": "user", "content": prompt})
    if st.session_state.chosen_id == 'ChatGPT+TechSensing':
        st.session_state.prompts3.append(prompt)
        st.session_state.messages3.append({"role": "user", "content": prompt})
    if st.session_state.chosen_id == 'ChatGPT+MyData':
        st.session_state.prompts4.append(prompt)
        st.session_state.messages4.append({"role": "user", "content": prompt})

with st.sidebar.expander("DOCUMENT"):
    st.session_state.multiple_files = st.file_uploader("Uploader", accept_multiple_files=True, key='file_uploader')
    btn_doc = st.button("SAVE", on_click = make_data_instance, key="btn_doc")  
with st.sidebar.expander("YOUTUBE"): 
    st.session_state.youtube_data = st.text_input("URL", key='youtubeurl', placeholder = 'insert youtube url')
    btn_youtube = st.button("SAVE", on_click = make_data_instance, key="btn_youtube")  
with st.sidebar.expander("WEB PAGE"): 
    st.session_state.single_page_data = st.text_input("URL", key = 'single_weburl') 
    btn_singlepage = st.button("SAVE", on_click = make_data_instance,  key="btn_singlepage")  

if st.session_state.chosen_id == "ChatGPT 3.5":
    col1_chat1, col2_chat1 = st.columns([6, 2])
    with col1_chat1.container(height=650, border= False):
        message_hist_display(st.session_state.messages1)
        if st.session_state.messages1[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):
                    prompt_ = convert_message(st.session_state.messages1)
                    response = st.session_state.llm.chat(prompt_) #결과                                            
                    chat_box(response.message.content)
                    message = {"role": "assistant", "content": response.message.content} #저장
                    st.session_state.messages1.append(message) # Add response to message history       
                res = websearch_func(prompt, response.message.content)
            if res:
                with st.chat_message("assistant", avatar = './src/web.png'):
                    message = {"role": "assistant", "content": res}
                    st.session_state.messages1.append(message) # Add response to message history           
                    chat_box(res)         

    with col2_chat1.container(height=650, border= False):    
        if (len(st.session_state.youtube_embeded_html) + len(st.session_state.img_embeded_html)) > 0:
            for i in st.session_state.youtube_embeded_html:
                st.write(f'''{i}<br>''',unsafe_allow_html=True)
            for i in st.session_state.img_embeded_html:
                st.write(f'''{i}<br>''',unsafe_allow_html=True)
        else:
            pass

if st.session_state.chosen_id == "ChatGPT 4":
    with st.container(height=650, border= False):
        message_hist_display(st.session_state.messages2)
        if st.session_state.messages2[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):
                    prompt_ = convert_message(st.session_state.messages2)
                    response = st.session_state.llm4.chat(prompt_) #결과                    
                    chat_box(response.message.content )                        
                    message = {"role": "assistant", "content": response.message.content} #저장
                    st.session_state.messages2.append(message) # Add response to message history       
                res = websearch_func(prompt, response.message.content)
            if res:
                with st.chat_message("assistant", avatar = './src/chatbot.png'):
                    message = {"role": "assistant", "content": res}
                    st.session_state.messages2.append(message) # Add response to message history           
                    chat_box(res)         
        
if st.session_state.chosen_id == "ChatGPT+TechSensing":  
    with st.container(height=650, border= False):
        message_hist_display(st.session_state.messages3)
        if st.session_state.messages3[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag.chat(prompt)
                        res = response.response
                    except Exception as e:
                        st.session_state.rag.reset()
                        res = str(e)
                    chat_box(res)
                    message = {"role": "assistant", "content": res}
                    st.session_state.messages3.append(message) # Add response to message history        
    
if st.session_state.chosen_id == "ChatGPT+MyData":
    col1_mychat, col2_mychat = st.columns([3, 2])
    with col1_mychat.container(height=650, border= False):
        message_hist_display(st.session_state.messages4)
        try:
            if st.session_state.messages4[-1]["role"] == "user":
                with st.chat_message("assistant", avatar = './src/chatbot.png'):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_engine2.chat(prompt)
                        chat_box(response.response)                        
                        message = {"role": "assistant", "content": response.response}
                        st.session_state.messages4.append(message) # Add response to message history        
                        
            if len(st.session_state.chat_db.summary) > 0:
                for i in st.session_state.chat_db.summary:
                    message = {"role": "assistant", "content": i}
                    st.session_state.messages4.append(message) # Add response to message history        
                    st.session_state.chat_db.summary = []
                    with st.chat_message("assistant", avatar = './src/chatbot.png'):
                        chat_box(i)
        except Exception as e:
            st.warning('Insert Documents | Youtube | Web Url', icon="⚠️")
            pass
    
    with col2_mychat.container(height=650, border= False):
        if len(st.session_state.display_datasource) > 0:
            display_customized_data(st.session_state.display_datasource)



# with st.sidebar.expander("WEB DEEP SEARCH"): 
#     all_page_data = st.text_input("URL", key = 'all_weburl') 
#     if st.session_state.webpageall_read_yn == False:
#         if len(all_page_data) > 0:
#             web_cracker = RemoteDepthReader()
#             web_crack_documents = web_cracker.load_data(url=all_page_data)
#             web_crack_documents = [i for i in web_crack_documents if i.split('.')[-1] == 'html']
#             with st.spinner("Fetching page texts"):
#                 docs = []
#                 for i in tqdm(web_crack_documents[:50]):
#                     loader = AsyncChromiumLoader([i])
#                     bs_transformer = BeautifulSoupTransformer()
#                     docs_transformed = bs_transformer.transform_documents(loader.load(), unwanted_tags= ['style','script'], tags_to_extract = ['a','p', 'span'])
#                     result_string = remove_nested_parentheses(docs_transformed[0].page_content)
#                     web_data_documents = [Document(text=result_string)]
#                     web_data_documents[0].metadata['resource'] = 'web_allpage'
#                     web_data_documents[0].metadata['title'] = 'web_allpage'
#                     docs.append(web_data_documents[0])
#             st.session_state.external_docs.append(docs)
#             st.session_state.webpageall_read_yn = True
#         else:
#             st.session_state.webpageall_read_yn = False
#     else:
#         pass
#     btn_allpage = st.button("SAVE", on_click = make_data_instance, args=[st.session_state.external_docs], key="btn_allpage")  