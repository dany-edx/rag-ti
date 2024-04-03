import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from io import StringIO, BytesIO
from tqdm.auto import tqdm
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
import llama_index.core
from llama_index.core.schema import Document
from llama_index.core import ServiceContext

import nest_asyncio
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append('/home/qcells/Desktop/rag_project/utils')
sys.path.append("/workspaces/hanwhaqcells/utils")
sys.path.append('../utils')
sys.path.append('./ui_utils')
from selenium_utils import global_obj, web_uploader, translated_func, youtube_uploader, websearch_func
from chatgpt_utils import set_llm, set_rag, set_llm4, set_embedding
from qcells_route_engine import qcell_engine, web_engine, high_level_engine
from qcells_custom_rag import create_db_chat, docx_load_data, pptx_load_data, pdf_load_data, get_timeline, generate_strategy
from streamlit_pdf_viewer import pdf_viewer
import phoenix as px
import os
from selenium import webdriver
from annotated_text import *
from htbuilder.units import unit
from collections import Counter
import asyncio
from httpx_oauth.clients.microsoft import MicrosoftGraphOAuth2
import threading 
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime import get_instance
import datetime
nest_asyncio.apply()
st.set_page_config(page_title="RAG",  layout="wide",  page_icon="☀️")
rem = unit.rem
parameters.LABEL_FONT_SIZE=rem(0.6)

#hanwha domain
# CLIENT_ID = 'f17632ac-7fc4-4525-a157-518f7cbcdc8d'
# CLIENT_SECRET = '3k48Q~HELekDJTlvz_vAAVXSSi-JoJshp~cPPc7z'
# REDIRECT_URI = 'https://qcells-us-rag.westus2.cloudapp.azure.com:442/'
# TENANT_ID = '0f7b4e1c-344e-4923-aaf0-6fca9e6700c8'

#us q-cells domain
CLIENT_ID = 'df709c24-e19e-4e77-b44a-f0b655304248'
CLIENT_SECRET = 'zRp8Q~07BqYLNTfzjOzGbkavXY~-53nugxqPqcPn'
REDIRECT_URI = 'https://qcells-us-rag.westus2.cloudapp.azure.com:442/'
TENANT_ID = '133df886-efe0-411c-a7af-73e5094bbe21'

@st.cache_resource
def lanch_px_app():
    llama_index.core.set_global_handler("arize_phoenix")
    px.launch_app(host='0.0.0.0')
lanch_px_app()

    
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
if "high_level_rag" not in st.session_state:
    st.session_state.high_level_rag = high_level_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
if "service_context" not in st.session_state:
    st.session_state.service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)
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
    st.session_state.chosen_id = 'ChatGPT+TechSensing'
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
if "display_datasource_idx" not in st.session_state.keys(): 
    st.session_state.display_datasource_idx = 0
if "is_done_translate" not in st.session_state.keys(): 
    st.session_state.is_done_translate = False
if "oauth_client" not in st.session_state.keys(): 
    st.session_state.oauth_client = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
if "is_signed_in" not in st.session_state.keys(): 
    st.session_state.is_signed_in = False
if "user_email" not in st.session_state.keys(): 
    st.session_state.user_email = ''
    
async def get_authorization_url(client: MicrosoftGraphOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(redirect_uri, scope=[CLIENT_ID + "/.default"])
    return authorization_url
    
def get_login_str():
    client: MicrosoftGraphOAuth2 = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
    authorization_url = asyncio.run(get_authorization_url(client, REDIRECT_URI))
    return authorization_url
    
async def get_email(token: str):
    user_id, user_email = await st.session_state.oauth_client.get_id_email(token)
    return user_id, user_email

async def get_access_token(code: str):
    token = await st.session_state.oauth_client.get_access_token(code, REDIRECT_URI)
    return token

def display_user():
    try:
        code = st.query_params.get_all('code')
        if code:
            st.session_state.is_signed_in = True
            # _, st.session_state.user_email = asyncio.run(get_email(st.session_state.oauth_client, token['access_token']))  
        else:
            st.session_state.is_signed_in = False
    except:
        pass
        
def make_data_instance():
    document_uploader()
    youtube_uploader()
    web_uploader()
    with st.spinner("Creating knowledge database"):
        st.session_state.chat_db = create_db_chat(st.session_state.external_docs, st.session_state.llm_rag, st.session_state.embedding,  st.session_state.service_context)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    st.session_state.chat_engine_react = ReActAgent.from_llm(st.session_state.chat_db.to_tool_list(), memory = memory, llm = st.session_state.llm_rag, verbose = True)    
    st.session_state.chat_engine2 = st.session_state.chat_db.multi_retriever()
    st.session_state.external_docs = []

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
            chat_box(msg)


def display_customized_data(_source):
    if 'pdf' in list(_source.keys())[0]:
        pdf_viewer(input=_source['pdf'], width=550)
    if 'youtube' in list(_source.keys())[0]:
        st.write('<iframe src="{}" width="100%" height="400px"></iframe>'.format(_source['youtube']),unsafe_allow_html=True,)
    if 'HTML' in list(_source.keys())[0]:
        st.markdown(_source['HTML'].replace('$', '\$'))
    if 'docx' in list(_source.keys())[0]:
        st.markdown(_source['docx'].replace('$', '\$'))
    if 'py' in list(_source.keys())[0]:
        st.code(_source['py'])

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
        st.session_state.webrag.reset()
        st.session_state.high_level_rag.reset()
        st.session_state.rag = qcell_engine(llm = st.session_state.llm_rag, embedding = st.session_state.embedding)
        st.session_state.messages3 = [{"role": "system", "content": "Hello, What can I do for you?"}]            
        st.session_state.prompts3 = []
    if x == 'ChatGPT+MyData':
        st.session_state.messages4 = [{"role": "system", "content": "Hello, What can I do for you?"}]    
        st.session_state.prompts4 = []
        st.session_state.display_datasource_idx = 0
    st.cache_resource.clear()
    st.session_state.youtubeurl = ''
    st.session_state.display_datasource = []
    
def document_uploader():        
    if len(st.session_state.multiple_files) > 0:
        st.session_state.external_docs = []
        for doc in st.session_state.multiple_files:
            doc_type = doc.name.split('.')[-1]
            if doc_type == 'pdf':
                string_data = pdf_load_data(doc)
                pdf_documents = [Document(text=string_data, metadata= {"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(pdf_documents)  
                st.session_state.display_datasource.append({doc_type: doc.getvalue()})
            if doc_type == 'py':
                stringio = StringIO(doc.getvalue().decode("utf-8"))
                string_data = stringio.read()
                document = [Document(text=string_data, metadata= {"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(document)
                st.session_state.display_datasource.append({doc_type: string_data})
            if doc_type == 'pptx':
                string_data = pptx_load_data(doc) 
                document = [Document(text=string_data, metadata={"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(document)    
                # st.session_state.display_datasource.append({doc_type: doc.getvalue()})
            if doc_type == 'docx':
                string_data = docx_load_data(doc) 
                document = [Document(text=string_data, metadata={"title" : doc.name, 'resource' : 'file'})]
                st.session_state.external_docs.append(document)
                st.session_state.display_datasource.append({doc_type: string_data})

def chat_box(text):
    texts = text.split('```')
    for i in texts:
        if i.split('\n')[0] ==  'python':
            with stylable_container(
                "codeblock2",
                """
                div {background-color:rgb(0,0,0,0.1); border-radius:10px} 
                code {font-size:11px;font-family: Arial;}
                """,):
                x = st.code(i, language = 'python', line_numbers = True)
        else:
            with stylable_container(
                "codeblock",
                """
                div {background-color:rgb(0,0,0,0)}
                code {font-size:11px; font-family: Arial; white-space: pre-wrap !important;}
                """,):
                x = st.code(i, language = 'md')
                
st.markdown('''
            <style>
                html {font-size:14px; font-family: Arial; padding-top: 15px}
                img {border-radius: 10px;}
                div[data-testid="stSidebarUserContent"]{style: "padding-top: 1rem";}
                div[data-testid="stExpander"]> details {border-width: 0px;} 
                div[data-testid="stCodeBlock"]> pre {background: rgb(248, 249, 255, 0);} 
                div[data-testid="stMarkdownContainer"]> p {font-size:11px; font-family: Arial;}
                button[data-testid="baseButton-secondary"] > p {font-size:11px; font-family: Arial;}
            </style>
            ''',unsafe_allow_html=True,)
st.sidebar.write('''<img width="200" height="60" src="https://us.qcells.com/wp-content/uploads/2023/06/qcells-logo.svg"  alt="Qcells"><br><br>''',unsafe_allow_html=True,)

display_user()
if st.session_state.is_signed_in == False:
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; URL={get_login_str()}">
    """, unsafe_allow_html=True)
    st.stop()
    
col1, col2  = st.columns([1, 5])
with col1:
    st.session_state.chosen_id = st.selectbox('', ('ChatGPT+TechSensing', 'ChatGPT 3.5', 'ChatGPT 4', 'ChatGPT+MyData'), label_visibility="collapsed", key = 'model_select')
with col2:
    with stylable_container(
        "reset1",
        """
        button {color: black;border-radius: 20px;}
        """,):
    
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
    
if st.session_state.chosen_id == "ChatGPT+MyData":
    with st.sidebar.expander("DOCUMENT"):
        st.session_state.multiple_files = st.file_uploader("Uploader", accept_multiple_files=True, key='file_uploader')
        translatable_docs = [m for m in st.session_state.multiple_files if m.name.split('.')[-1] != 'py']
        if len(st.session_state.multiple_files) > 0:
            btn_doc = st.button("START TALK", on_click = make_data_instance, key="btn_doc", use_container_width=True)  
            st.divider()
        sub_btn_col1, sub_btn_col2 =st.columns(2)
        with sub_btn_col1:
            if len(translatable_docs)>0:
                lang_selector = st.selectbox('', ('en', 'ko'), label_visibility="collapsed", key = 'lang_selector')
        with sub_btn_col2:
            if len(translatable_docs)>0:
                btn_trans = st.button("TRANSLATE", on_click = translated_func, key="translate", use_container_width=True)
            else:
                st.session_state.is_done_translate = False
        if len(translatable_docs)>0:
            if st.session_state.is_done_translate == True:
                for trans_doc in translatable_docs:
                    with open('../tmp/translated/' + 'translated_'+ trans_doc.name, "rb") as file:
                        file_name = 'translated_' + trans_doc.name
                        btn = st.download_button(label=file_name, data=file, file_name=file_name, use_container_width=True)
    
    with st.sidebar.expander("YOUTUBE"): 
        st.session_state.youtube_data = st.text_input("URL", key='youtubeurl', placeholder = 'insert youtube url')
        if len(st.session_state.youtube_data) > 0:
            btn_youtube = st.button("START TALK", on_click = make_data_instance, key="btn_youtube", use_container_width=True)  
    with st.sidebar.expander("WEB PAGE"): 
        st.session_state.single_page_data = st.text_input("URL", key = 'single_weburl', placeholder = 'insert html url')
        if len(st.session_state.single_page_data) > 0:
            btn_singlepage = st.button("START TALK", on_click = make_data_instance,  key="btn_singlepage",use_container_width=True)  

if st.session_state.chosen_id == "ChatGPT 3.5":
    with st.sidebar:
        st.header('Tutorial prompt')
        with st.expander("Example Q1. Simple Questions"):
            st.write('- What is Hanwha Qcells business?')
        with st.expander("Example Q2. Google browsing"):
            st.write('- What is the current weather in san francisco?')
            st.write('- Please find tesla stock price now.')
            st.write('- Please let me know about IQ8 Microinverter price')
        with st.expander("Example Q3. Coding correction"):
            st.write('''- What is the current weather in san francisco?''')

    annotated_text(
        "", annotation("ChatGPT3.5", "Function", font_size="0.7rem"),
        "", annotation("Google search", "Function",  font_size="0.7rem"),
    )    
    col1_chat1, col2_chat1 = st.columns([6, 2])
    with col1_chat1.container(height=650, border= False):
        message_hist_display(st.session_state.messages1)
        if st.session_state.messages1[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):                    
                    prompt_ = [ChatMessage(role=i['role'], content=i['content']) for i in st.session_state.messages1]
                    response =   st.session_state.llm.chat(prompt_) #결과                   
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
    annotated_text(
        "", annotation("ChatGPT4", "Function", font_size="0.7rem"),
        "", annotation("Google search", "Function",  font_size="0.7rem"),
    )

    with st.container(height=650, border= False):
        message_hist_display(st.session_state.messages2)
        if st.session_state.messages2[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):
                    prompt_ = [ChatMessage(role=i['role'], content=i['content']) for i in st.session_state.messages2]
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
    with st.sidebar:
        st.header('Tutorial prompt')
        with st.expander("Example Q1. Paper search"):
            st.write('- Please search object detection deep learning papers.')
            st.write('- I would like to talk about the second paper.')
            st.write("- Let's talk about the pdf. please summarize.")
            st.write("- Answer based on pdf. what is the key points of the paper? and explain about algorithm")
        with st.expander("Example Q2. Patent search"):
            st.write('- Find Virtual power plant patent by enphase energy.')
            st.write('- I would like to talk about the first patent.')
            st.write("- Answer based on pdf. make a report about including key technologies in the patent.")
        with st.expander("Example Q3. Product news search"):
            st.write('- what is the latest released product names by enphase energy in 2024?')
            st.write('- please provide 5 bullet points based on the pdf.')
            st.write('- add emoji')
            st.write("- let's talk about https://www.youtube.com/watch?v=YnyykZ8O1Eo&t=237s")
            st.write("- find comparison categories based on the youtube video.")
            st.write("- make comparison markdown table based on youtube video.")
        with st.expander("Example Q4. Direct Url talk"):
            st.write('- extract pdf. https://www.nature.com/articles/s41467-024-46334-4.pdf')
            st.write('- please provide 5 bullet points based on the pdf.')
            st.write('- add emoji')
            st.write("- let's talk about https://www.youtube.com/watch?v=YnyykZ8O1Eo&t=237s")
            st.write("- find comparison categories based on the youtube video.")
            st.write("- make comparison markdown table based on youtube video.")
        with st.expander("Example Q5. News search (with high level query)"):
            st.write('- Please find newly released product name by enphase energy in 2023, 2024.')
            st.write('- find each product prices on google searching and make a report using markdown table.')
        with st.expander("Example Q6. News search v2 (with high level query)"):
            st.write('- find 2024 top-5 largest solar panel manufacturing capacity company in USA')
            st.write('- find each company manufacturing and business plan in 2024.')
        
    on = st.toggle('High level Query')
    annotation_size = '0.8rem'
    annotated_text(
        "", annotation("google.com", "Search", font_size=annotation_size, background='#ffe6e6'),
        "", annotation("justia.com", "Patents", font_size=annotation_size),
        "", annotation("google.patent.com", "Patents",  font_size=annotation_size),
        "", annotation("paperswithcode.com", "Papers", font_size=annotation_size),
        "", annotation("nature.com", "Papers",  font_size=annotation_size),
        "", annotation("ercot.com", "Operation",  font_size=annotation_size),
        "", annotation("caiso.com", "Operation",  font_size=annotation_size),
        "", annotation("cpuc.ca.gov", "Commission",  font_size=annotation_size),
        "", annotation("puc.texas.gov", "Commission",  font_size=annotation_size),
        "", annotation("aws.amazon.com", "Manual",  font_size=annotation_size),
        "", annotation("microsoft.com", "Manual",  font_size=annotation_size),
        "", annotation("developer.ibm.com", "Manual",  font_size=annotation_size),
        "", annotation("pv-magazine.com", "News",  font_size=annotation_size),     
        "", annotation("reddit.com", "SNS",  font_size=annotation_size, background='#D38CAD'),
        "", annotation("PDF", "Function", font_size=annotation_size),
        "", annotation("PPTX", "Function",  font_size=annotation_size),
        "", annotation("DOCX", "Function",  font_size=annotation_size),
        "", annotation("YOUTUBE", "Function",  font_size=annotation_size),
        "", annotation("HTML", "Function",  font_size=annotation_size),
    )

    with st.container(height=620, border= False):
        message_hist_display(st.session_state.messages3)
        if st.session_state.messages3[-1]["role"] == "user":
            with st.chat_message("assistant", avatar = './src/chatbot.png'):
                with st.spinner("Thinking..."):
                    try:
                        if on == False:
                            response = st.session_state.rag.chat(prompt)
                        if on == True:
                            response = st.session_state.high_level_rag.chat(prompt) 
                            # st.session_state.high_level_rag.reset()
                        res = response.response
                    except Exception as e:
                        st.session_state.rag.reset()
                        res = str(e)
                    chat_box(res)
                    message = {"role": "assistant", "content": res}
                    st.session_state.messages3.append(message) # Add response to message history        


def next_material_page(func):
    if func == 'next':
        index_int = 1
    if func == 'before':
        index_int = -1
    if len(st.session_state.display_datasource)-1 > st.session_state.display_datasource_idx:
        st.session_state.display_datasource_idx = st.session_state.display_datasource_idx + index_int
    else:
        st.session_state.display_datasource_idx = 0
    print(st.session_state.display_datasource_idx)

if st.session_state.chosen_id == "ChatGPT+MyData":
    keys = [list(d.keys())[0] for d in st.session_state.display_datasource]
    key_counts = Counter(keys)    
    annotated_text(
        "", annotation("PDF", str(key_counts['pdf']), font_size="0.7rem"),
        "", annotation("PPTX", str(key_counts['pptx']), font_size="0.7rem"),
        "", annotation("DOCX", str(key_counts['docx']), font_size="0.7rem"),
        "", annotation("YOUTUBE", str(key_counts['youtube']),  font_size="0.7rem"),
        "", annotation("WEB", str(key_counts['HTML']),  font_size="0.7rem"),
        "", annotation("Translate", "Function", font_size="0.7rem"),
    )
    col1_mychat, col2_mychat = st.columns([3, 2])
    with col1_mychat.container(height=650, border= False):
        message_hist_display(st.session_state.messages4)
        try:
            if len(st.session_state.chat_db.summary) > 0:
                for i in st.session_state.chat_db.summary:
                    message = {"role": "assistant", "content": i}
                    st.session_state.messages4.append(message) # Add response to message history        
                    st.session_state.chat_db.summary = []
                    with st.chat_message("assistant", avatar = './src/chatbot.png'):
                        chat_box(i)
                st.rerun()
            
            if st.session_state.messages4[-1]["role"] == "user":
                with st.chat_message("assistant", avatar = './src/chatbot.png'):
                    with st.spinner("Thinking..."):
                        if len(st.session_state.chat_db) == 1:
                            # response = st.session_state.chat_engine_bm.query(prompt)
                            response = st.session_state.chat_engine_react.chat(prompt, tool_choice = 'hybrid_retriever_documents')
                        else:
                            response = st.session_state.chat_engine2.chat(prompt)
                        chat_box(response.response)                        
                        message = {"role": "assistant", "content": response.response}
                        st.session_state.messages4.append(message) # Add response to message history        
        
        except Exception as e:
            st.warning('Insert Documents | Youtube | Web Url', icon="⚠️")
            pass

    with col2_mychat.container(height=650, border= False):
        col1_btn, col2_btn = st.columns([1,9])
        if len(st.session_state.display_datasource) > 0:
            if len(st.session_state.display_datasource) > 1:
                with col1_btn:
                    st.button("⬅️", on_click = next_material_page, args=["before"], key="btn_before_page")
                with col2_btn:
                    st.button("➡️", on_click = next_material_page, args=["next"], key="btn_next_page")                
            display_customized_data(st.session_state.display_datasource[st.session_state.display_datasource_idx])

