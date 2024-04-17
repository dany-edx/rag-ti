import streamlit as st
import pandas as pd
import numpy as np
import glob
import datetime 
from llama_index.llms.azure_openai import AzureOpenAI
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie
from streamlit_extras.stylable_container import stylable_container
from llama_index.core.agent import ReActAgent
from streamlit_echarts import st_pyecharts
st.set_page_config(page_title="LOGGER",  layout="wide", page_icon="☀️")
st.sidebar.write('''<img width="200" height="60" src="https://us.qcells.com/wp-content/uploads/2023/06/qcells-logo.svg"  alt="Qcells"><br><br>''',unsafe_allow_html=True,)

def cnt_lines(file_path):
    with open(file_path, 'r') as file:
        all_lines = [i for i in file.readlines() if i.split('\t')[-1] == 'access\n']
    return len(all_lines)


def chat_box(text):
    texts = text.split('```')
    for i in texts:
        if i.split('\n')[0].lower() in ['java', 'javascript', 'python']:
            with stylable_container(
                "codeblock2",
                """
                div {background-color:rgb(0,0,0,0.1); border-radius:10px} 
                code {font-size:11px;font-family: Arial;}
                """,):
                x = st.code(i, language = i.split('\n')[0], line_numbers = True)
        else:
            with stylable_container(
                "codeblock",
                """
                div {background-color:rgb(0,0,0,0)}
                code {font-size:11px; font-family: Arial; white-space: pre-wrap !important;}
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
            chat_box(msg)
            
today_date = datetime.datetime.now().strftime('%Y-%m-%d')
yester_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

file_list = glob.glob('../ui/prompt_text/*.log')
file_list = sorted(file_list)

text_merge = []
for f in file_list[:7]:
    with open(f, 'r') as file:
        all_lines = file.readlines() 
        text_merge = text_merge + all_lines

df = pd.DataFrame([i.split('\t') for i in text_merge], columns = ['log_time','type1','type2','email','dummy'])
df['YYYYMMDD'] = df['log_time'].apply(lambda x:x[:10])
df['log_time'] = pd.to_datetime(df['log_time'])
df = df.set_index(['log_time'])

df_user_access = df[(df['type2'] == 'INFO') & (df['dummy'] == 'access\n') & (df['YYYYMMDD'] == today_date)]
df_chat_count = df[(df['type2'] == 'DEBUG')& (df['YYYYMMDD'] == today_date)]
df_trans_access = df[(df['type2'] == 'INFO') & (df['dummy'] == 'translate\n') & (df['YYYYMMDD'] == today_date)]

file_list_cnt = [cnt_lines(i) for i in file_list[:7]]
file_list_name = [i.split('/')[-1] for i in file_list[:7]]

col0, col1, col2 =st.columns(3)
with col0:    
    b = (
        Line()
        .add_xaxis(file_list_name)
        .add_yaxis("Count", file_list_cnt)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Daily access trend", subtitle=""),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical")
        )
    )
    st_pyecharts(b)
    
with col1:
    cnt_oneday_lag = len(df[(df['type2'] == 'INFO') & (df['dummy'] == 'access\n') & (df['YYYYMMDD'] == yester_date)])
    st.metric("User access count", "{} visitors".format(len(df_user_access)), "{}".format(len(df_user_access) - cnt_oneday_lag))
    with st.expander("show more data"):
        st.dataframe(df_user_access[['email']].reset_index())
with col2:
    cnt_oneday_lag = len(df[(df['type2'] == 'DEBUG')& (df['YYYYMMDD'] == yester_date)])
    st.metric("User chat count", "{} chats".format(len(df_chat_count)), "{}".format(len(df_chat_count) - cnt_oneday_lag))
    with st.expander("show more data"):
        st.write(df_chat_count[['email']].reset_index())

df_user_access_gp = df_user_access.groupby(['email']).count().reset_index()
df_chat_count_gp = df_chat_count.groupby(['email']).count().reset_index()
df_trans_access_gp = df_trans_access.groupby(['email']).count().reset_index()
df_chat_count_gp_by_model = df_chat_count.groupby(['dummy']).count().reset_index()

col0, col1, col2 =st.columns(3)
with col0:
    b = (
        Bar()
        .add_xaxis(df_user_access_gp.email.tolist())
        .add_yaxis("Count", df_user_access_gp.type1.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Steady user?", subtitle=""),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45,font_size=8)),
        )
    )
    
    st_pyecharts(b)

with col1:
    b = (
        Bar()
        .add_xaxis(df_chat_count_gp.email.tolist())
        .add_yaxis("Count", df_chat_count_gp.type1.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Steady chatter?", subtitle=""),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45,font_size=8)),
        )
    )
    
    st_pyecharts(b)

with col2:
    b = (
        Bar()
        .add_xaxis(df_trans_access_gp.email.tolist())
        .add_yaxis("Count", df_trans_access_gp.type1.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Steady translate user?", subtitle=""),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45,font_size=8)),
        )
    )
    st_pyecharts(b)

col0, col1, col3 =st.columns(3)
with col0:
    b = (
        Pie()
        .add(series_name = 'model name',data_pair = df[df['type2'] == 'DEBUG'].groupby(['dummy']).count().reset_index()[['dummy', 'type1']].values.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Steady model?", subtitle=""),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
        )
        
    )
    st_pyecharts(b)



# with col1:            
#     llm = AzureOpenAI(
#                 model="gpt-35-turbo",
#                 deployment_name="qcell_gpt_model",
#                 temperature = 0,
#                 api_key="c11ed4df2d35412b89a7b51a631bf0e4",
#                 azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
#                 api_version="2024-02-15-preview")
    
    
#     messages1 = [{"role": "system", "content": "Hello, What can I do for you?"}]    
#     message_hist_display(messages1)
#     if messages1[-1]["role"] == "user":
#         with st.chat_message("assistant", avatar = './src/chatbot.png'):
#             with st.spinner("Thinking..."):                    
#                 prompt_ = [ChatMessage(role=i['role'], content=i['content']) for i in messages1]
#                 response = llm.chat(prompt_) #결과                   
#                 chat_box(response.message.content)
#                 message = {"role": "assistant", "content": response.message.content} #저장
#                 messages1.append(message) # Add response to message history  
#     if prompt := st.chat_input("Your question", key = 'chat_input_query'): # Prompt for user input and save to chat history
#         prompts1.append(prompt)
#         messages1.append({"role": "user", "content": prompt})
