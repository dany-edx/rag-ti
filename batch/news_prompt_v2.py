import os 
import wikipedia
from llama_index.core import download_loader
import glob
import pandas as pd
from datetime import datetime, timedelta
from llama_index.core import ServiceContext
from functools import wraps
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.llms import ChatMessage
import sys
sys.path.append('../utils')
from db_utils import * 
du = DB_Utils()
from llama_index.core import PromptTemplate
from llama_index.core import SummaryIndex
from llama_index.core import PromptTemplate
from pydantic import Field, BaseModel
from llama_index.core import SummaryIndex, get_response_synthesizer, VectorStoreIndex, Document, KeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import SummaryIndex, get_response_synthesizer, StorageContext, load_index_from_storage, VectorStoreIndex, set_global_service_context, SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, Document

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="qcell_gpt_model",
    temperature = 0,
    api_key="c11ed4df2d35412b89a7b51a631bf0e4",
    azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
    api_version="2024-02-15-preview",
    # max_tokens=4096
)

embedding = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="qcell_embedding_model",
    api_key="c11ed4df2d35412b89a7b51a631bf0e4",
    azure_endpoint="https://rag-openai-qcells-east.openai.azure.com/",
 api_version="2023-07-01-preview",
)

service_context = ServiceContext.from_defaults(llm=llm,embed_model=embedding)
set_global_service_context(service_context)

def get_raw_news_data():
    df_merge = du.fetch_data(sql = '''select * from new_released_data''')    
    return df_merge

class Text_answer(BaseModel):
    """
        Restructure answers.    
        Return:
            Position: 
            relavant score:
            Reason: 
            Keywords: Keywords and emoji
    """
    
    Position: str
    relavant_score: float
    Reason: str     
    keywords: str

def iterative_prompt(df_merge):
    data = []
    for i in range(len(df_merge)):
        try:
            documents = Document(text=df_merge.iloc[i].News_Contents)
            index = GPTVectorStoreIndex.from_documents([documents])
            response_synthesizer = get_response_synthesizer(text_qa_template=map_template, 
                                                            response_mode="refine", 
                                                            output_cls = Text_answer)
            
            query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, use_async=True, verbose = True)
            url = df_merge.iloc[i].News_Url
            title = df_merge.iloc[i].News_Title
             
            for i in job_titles:
                response = query_engine.query(i)
                print(response.response)
                data.append([response.response.Position, response.response.relavant_score, response.response.keywords, response.response.Reason, url, title])
        except:
            pass
            
    result_df = pd.DataFrame(data, columns = ['position', 'relavance_score', 'keywords', 'Summary', 'url', 'title'])
    result_df['position'] = result_df['position'].apply(lambda x:x.split(':')[0])
    return result_df


def html_merge(html_element, position_name):
    html = f'''
    <tr>
        <td style="padding: 20px;">
            <hr>
            <a style="color: #000000; text-decoration: none; display: block; font-size: 12px;">{position_name}</a>
        </td>
    </tr>
    {html_element}
    '''
    return html


def html_format(url, title, relavance_score,keywords, summary):
    summary = summary.replace('\n', '<br>')
    html = f'''
    <tr>
        <td style="padding: 20px; text-align: left;">
            <div style="word-wrap: break-word; font-size: 12px;">
                <a href = {url}>
                {title}
                </a>
                <a>
                    <br> releavance score: {relavance_score}<br>
                    <br> Reason: {summary}<br>
                    <br> keywords: {keywords}<br>        
                    
                </a>
            </div>
        </td>
    </tr>
    '''
    
    html = html
    return html


def parse_to_html(result_df):
    html_merged = ''
    for position_name in result_df['position'].unique():
        result_df_ = result_df[(result_df['relavance_score']>=0.7)]
        result_df_ = result_df_[result_df_['position'] == position_name]
        result_df_ = result_df_.sort_values(by = 'relavance_score', ascending = False)
        result_df_ = result_df_.head(3)
        if len(result_df_) > 0:
            result_df_['html'] = result_df_.apply(lambda x: html_format(x['url'], x['title'], x['relavance_score'], x['keywords'], x['Summary']), axis = 1)
            result_df_ = result_df_.sort_values(by = 'relavance_score', ascending = False).head(3)
            htmls = result_df_['html'].values
            html_merged = html_merged + html_merge('\n'.join(htmls[:3]), position_name)
    return html_merged

def save_html(html_merged):
    with open("../tmp/base_template.html", "r", encoding='utf-8') as f:
        text = f.read()
    text = text.replace('datainput', html_merged)
    with open("../data/output/news_template.html", "w", encoding='utf-8') as f:
        f.write(text)



map_template = PromptTemplate(   
    "The following is a the given Position and description.\n"
    "{query_str}\n\n"
    "The following is a content from the news article.\n"
    "{context_str}\n\n"

    "The following is the instruction\n"
    "1.Evaluate direct and technical relevance score(0 to 1 as float) between the the given Position description and the news content. 0 is the lowest relevance, 1 is the highest relevance. \n"
    "2.Write a reason of the score. 200 characters limited. \n"
    "3.Write only three keywords(must add each keyword with an emoji) from content.\n."
)

job_titles = [
    " Software Developer Backend Engineer : To oversee Cloud-side development for EMS-Cloud TCP/MQTT/HTTPS communications, API development for data handling in web/apps, and firmware update controls. Utilizing Java/Spring Boot, I focus on internal communication through Kafka and HTTPS, manage ELK stack logging, and assess alternative tech stacks for efficiency. My responsibilities include AWS MSK and OpenSearch monitoring, architecture design for optimal maintenance and performance, and direct partnership collaboration. I aim to deepen my expertise in Kafka, Java, Spring Boot, broaden my backend technology scope, and improve technical communication to effectively convey complex concepts to varied audiences."
    ,"Software Developer DevOps Engineer: basic programming experience in C and C++, along with experience in setting up Linux servers, software development, an understanding of the software development life cycle, Docker, knowledge of CI/CD processes, experience with Jenkins and the Yocto build system, AWS services, and proficiency in JIRA/Confluence. My areas of interest include the overall management of software, from its creation, building, deployment, to management."
    ,"PV Technology Strategy : This role focuses on R&D in solar cell and module technology, involving competitive technology analysis, development of technology roadmaps, adoption and application of new technological trends, understanding technology development trends across the entire value chain, and enhancing technological competitiveness through intellectual property review. The goal is to ensure the company's technology development pace surpasses competitors by swiftly integrating global innovations and trends."
    ,"Software Technology Strategy : This position entails crafting technology roadmaps for PV, focusing on TOPCon and Tandem technologies, utilizing intelligence for strategic R&D planning, and revising IP strategies to pinpoint patenting opportunities and address risks. Duties also include competitive analysis, roadmap development for immediate and future tech applications, and report preparation."
    ,"Legal Managers : Oversee Only Hanwha Qcells related legal compliance and risk management, providing expert legal counsel on corporate law, intellectual property, and regulatory matters, aligning legal strategies with Hanwha QCELLS' business objectives in energy, chemicals, and real estate."
]

def news_main():
    df_merge = get_raw_news_data()
    result_df = iterative_prompt(df_merge)
    result_df = result_df.rename(columns = {'index' : 'position'})
    # result_df = du.fetch_data('''select * from pv_magazine pm where Released_Date = (select max(Released_Date) from pv_magazine)''')
    html_merged = parse_to_html(result_df)
    save_html(html_merged)
    du.insert_pd_tosql('news_output', result_df)


if __name__ == '__main__':
    news_main()
