import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
import pandas as pd
import os 
import wikipedia
from llama_index.core import download_loader
from random import randint
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.memory import ChatMessageHistory
__import__('pysqlite3')
import sys
sys.path.append('../batch')
sys.path.append('../utils')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from news_prompt import *
from qcells_route_engine import *
from llama_index.vector_stores.chroma import ChromaVectorStore
import sys
sys.path.append('../utils')
from db_utils import * 
du = DB_Utils()

wiki_loader = download_loader("WikipediaReader", custom_path='./wikipedia')
wiki_loader = wiki_loader()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'

st.session_state.embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")
st.session_state.llm = AzureChatOpenAI(temperature = 0, deployment_name="test_gpt")
service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]
    
if "selection" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.selection = pd.DataFrame()

if "firstChatYN" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.firstChatYN = True

if "AnalyYN" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.AnalyYN = False

# if "analysis_output" not in st.session_state.keys(): # Initialize the chat message history
#     st.session_state.analysis_output1 = None

# if "analysis_output2" not in st.session_state.keys(): # Initialize the chat message history
#     st.session_state.analysis_output2 = None

# if "analysis_output3" not in st.session_state.keys(): # Initialize the chat message history
#     st.session_state.analysis_output3 = pd.DataFrame()

@st.cache_data
def get_today_data():
    df = du.fetch_data(sql = '''select * from pv_magazine pm where Released_Date = (select max(Released_Date) from pv_magazine)''')    

    # search_datetime = datetime.strftime(datetime.now()- timedelta(days = 1),'%Y%m%d')
    # df = pd.read_csv('../data/output/{}/news_result.csv'.format(search_datetime))  
    # # st.write(df)
    return df

@st.cache_resource(show_spinner=True)
def load_chat_engine():
    chat_engine = qcell_engine()
    return chat_engine


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections = df_with_selections.drop_duplicates(subset=['title'])
    df_with_selections.insert(0, "ID", False)    
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        disabled=df.columns,
        column_order=("ID", "title","impact", "impact_summary"),
        on_change=reset_dataframe_selector
    )
    
    selected_rows = edited_df[edited_df.ID]
    if len(selected_rows) >= 1:     
        st.session_state.selection = selected_rows[-1:]
    else:
        pass

def reset_dataframe_selector():
    st.session_state.AnalyYN = False

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]
    st.session_state.chat_engine.reset()
    st.session_state.firstChatYN = True

# def get_semantic_analysis(texts):
#     _, st.session_state.analysis_output1, st.session_state.analysis_output2 = org_news_to_summary(texts)
#     df = pd.DataFrame.from_dict(st.session_state.analysis_output1, orient='index', columns=['Relevance_Score', 'Interesting_Score', 'Summary'])
#     df.index.name = 'Job_Title'
#     df.reset_index(inplace=True)
#     st.session_state.analysis_output3 = df
    

st.set_page_config(layout="wide", page_title="NEW ANALYSIS", page_icon="📰")
st.markdown("# News Analysis powered by LLM")
st.sidebar.header("RAG Demo")

df = get_today_data()


st.session_state.chat_engine = load_chat_engine()

# start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=1)  #  I need some range in the past
# end_date = dt.datetime.now().date()-relativedelta(years=0)

with st.sidebar.form(key='Search'):
    # slider = st.slider('New release period', min_value=start_date, value=(start_date ,end_date) ,max_value=end_date, format='YYYY MMM DD', )
    # countries = st.multiselect("Choose countries", df.news_national_type.drop_duplicates().values.tolist(), [])
    text_query = st.text_input(label='Enter text to search')
    submit_button = st.form_submit_button(label='Search')
    # if countries:
    #     df = df[df['news_national_type'].isin(countries)]
    # if text_query:
    #     df = search_dataframe(df, "contents", text_query)
    # df_filterd = df[(pd.to_datetime(df['release_date']).dt.date>=slider[0]) & (pd.to_datetime(df['release_date']).dt.date<=slider[1])]

col1, col2 = st.columns([1, 1])
with col1:
    st.write('News dataset')
    dataframe_with_selections(df)
    
with col2:
    st.write('General ChatAgent')
    st.button('Reset Chat', on_click=reset_conversation)
    container = st.container(height=300)    
    if prompt := st.chat_input("Say something"):
        if st.session_state.firstChatYN == True:
            if len(st.session_state.selection)>0:
                prompt = prompt + "\n\ndocument title: " + st.session_state.selection.title.values[0]
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.firstChatYN = False
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})                    
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        container.chat_message(message['role']).write(message['content'])
    
    if st.session_state.messages[-1]["role"] != "assistant":
        response = st.session_state.chat_engine.chat(prompt)
        container.chat_message("assistant").write(f"Echo: {response.response}")
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message) # Add response to message history

if len(st.session_state.selection) > 0:
    with st.expander("News info"): 
        st.subheader(st.session_state.selection.title.values[0])
        st.write(st.session_state.selection.url.values[0])
    with st.expander("News contents"): 
        st.write(st.session_state.selection.impact.values[0])
        st.write(st.session_state.selection.impact_summary.values[0])
        st.write(st.session_state.selection.impact_analogy.values[0])
    
    # if st.session_state.AnalyYN == False:
        # get_semantic_analysis(st.session_state.selection.contents.values[0])
        # for message in st.session_state.messages: # Display the prior chat messages
        #     container.chat_message(message['role']).write(message['content'])
        # st.session_state.AnalyYN = True

#     with st.expander("Analyzed"):    
#         st.write('Hanwha QCELLS Impact: ', st.session_state.analysis_output2[0])
#         st.write('Summary: ', st.session_state.analysis_output2[1])
#         st.data_editor(
#             st.session_state.analysis_output3,
#             column_config={
#                 "Interesting_Score": st.column_config.ProgressColumn("Interesting_Score",format="%f",min_value=0,max_value=1),
#                 "Relevance_Score": st.column_config.ProgressColumn("Relevance_Score",format="%f",min_value=0,max_value=1),
#             },
#             hide_index=True,
#             use_container_width = True
#         )
#     st.success('Done!')
