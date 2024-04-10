from selenium.webdriver.chrome.options import Options
import streamlit as st
from llama_index.core import get_response_synthesizer,  ServiceContext, Document
from qcells_route_engine import qcell_engine, web_engine, Decide_to_search_external_web, GoogleRandomSearchToolSpec, VectordbSearchToolSpec, high_level_engine
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from qcells_custom_rag import create_db_chat, docx_load_data, pptx_load_data, pdf_load_data, get_youtube_metadata, get_timeline, generate_strategy
from youtube_transcript_api import YouTubeTranscriptApi
import os
from azure_translate import SampleTranslationWithAzureBlob
from qcells_web_instance_search import instance_search_expanding

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

def heartbeat():
    uuid = px.Client().get_trace_dataset().save(directory='./prompt_text')
    with open("writelog.log", 'a') as f:
        f.write(f"Alive at {datetime.datetime.now()}\n")

def start_beating():
    thread = threading.Timer(interval=2, function=start_beating)
    add_script_run_ctx(thread)
    ctx = get_script_run_ctx()     
    runtime = get_instance()     # this is the main runtime, contains all the sessions    
    if runtime.is_active_session(session_id=ctx.session_id):
        thread.start()
    else:
        heartbeat()
        return

def get_answer_yn(llm, query_str, text_chunks):
    synthesizer = get_response_synthesizer(llm = llm, response_mode="refine", output_cls = Decide_to_search_external_web)
    result_response = synthesizer.get_response(query_str = query_str, text_chunks=[text_chunks], verbose = True)      
    return result_response


def websearch_func(prompt, response):
    answer_eval = get_answer_yn(st.session_state.llm, prompt, response)
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

def translated_func():
    if len(st.session_state.multiple_files) > 0:        
        for trans_doc in st.session_state.multiple_files:
            with st.spinner('[{}]'.format(trans_doc.name) + " translating..."):
                with open(os.path.join("../tmp/translated/" + trans_doc.name),"wb") as f:
                    f.write(trans_doc.getbuffer())
                sample = SampleTranslationWithAzureBlob()
                poller = sample.sample_translation_with_azure_blob(trans_doc.name, to_lang = st.session_state.lang_selector)
        st.session_state.is_done_translate = True
    else:
        st.toast('Need to insert document!')


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