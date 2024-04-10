from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import streamlit as st
from annotated_text import *
from collections import Counter
from qcells_route_engine import qcell_engine, web_engine, high_level_engine
from llama_index.core import ServiceContext
import asyncio
from httpx_oauth.clients.microsoft import MicrosoftGraphOAuth2
#us q-cells domain

CLIENT_ID = 'df709c24-e19e-4e77-b44a-f0b655304248'
CLIENT_SECRET = 'zRp8Q~07BqYLNTfzjOzGbkavXY~-53nugxqPqcPn'
REDIRECT_URI = 'https://qcells-us-rag.westus2.cloudapp.azure.com:442/login'
TENANT_ID = '133df886-efe0-411c-a7af-73e5094bbe21'

#hanwha domain
CLIENT_ID2 = 'f17632ac-7fc4-4525-a157-518f7cbcdc8d'
CLIENT_SECRET2 = '3k48Q~HELekDJTlvz_vAAVXSSi-JoJshp~cPPc7z'
REDIRECT_URI2 = 'https://qcells-us-rag.westus2.cloudapp.azure.com:442/login-hanwha'
TENANT_ID2 = '0f7b4e1c-344e-4923-aaf0-6fca9e6700c8'            

async def get_authorization_url(client: MicrosoftGraphOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(redirect_uri, scope=[CLIENT_ID + "/.default"])
    return authorization_url
    
def get_login_str():
    client: MicrosoftGraphOAuth2 = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
    authorization_url = asyncio.run(get_authorization_url(client, REDIRECT_URI))
    return authorization_url

async def get_authorization_url_hanwha(client: MicrosoftGraphOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(redirect_uri, scope=[CLIENT_ID2 + "/.default"])
    return authorization_url
    
def get_login_str_hanwha():
    client: MicrosoftGraphOAuth2 = MicrosoftGraphOAuth2(CLIENT_ID2, CLIENT_SECRET2)
    authorization_url = asyncio.run(get_authorization_url_hanwha(client, REDIRECT_URI2))
    return authorization_url


async def get_email(client: MicrosoftGraphOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email
    
def get_use():
    client = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
    code = st.query_params.get_all('code')
    token = asyncio.run(get_access_token(client, REDIRECT_URI, code))
    user_id, user_email = asyncio.run(get_email(client, token['access_token']))
    st.write(f"You're logged in as {user_email} and id is {user_id}")

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

def get_tutorial_gpt():
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
    
def get_anno_tech_sensing():
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

def get_anno_mydata():
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

def get_anno_mydata():
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
    
def get_tutorial_tech_sensing():
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

def get_session_init():
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
    if "is_signed_in" not in st.session_state.keys(): 
        st.session_state.is_signed_in = False
    if "user_email" not in st.session_state.keys(): 
        st.session_state.user_email = ''
    if "access_token" not in st.session_state.keys(): 
        st.session_state.access_token = ''
    if "access_mail" not in st.session_state.keys(): 
        st.session_state.access_mail = ''
    if "refresh_token" not in st.session_state.keys(): 
        st.session_state.refresh_token = ''
    if "saved_log" not in st.session_state.keys(): 
        st.session_state.saved_log = False