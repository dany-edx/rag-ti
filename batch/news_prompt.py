import os 
import wikipedia
from llama_index.core import download_loader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory
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

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'

# embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="test_gpt",
    temperature = 0,
    api_key="70d67d8dd17f436b9c1b4e38d2558d50",
    azure_endpoint="https://qcells-us-test-openai.openai.azure.com/",
    api_version="2023-07-01-preview",
)

# service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding,)

wiki_loader = download_loader("WikipediaReader")
wiki_loader = wiki_loader()

company_desc = wikipedia.summary("q_cells", sentences=5)
template = f"""
You are an news remmcomandation assistant who currently employed at "Hanwha Solution Corporation". 
{company_desc}
"""


jsonfy_txt = '''
When I provide an news article, you evaluate the distinct and clear relevance or interest based on each position explanation.
Additionally, Create summary(without subject) using keywords from the article for each position.

===========Positions explanations at Hanwha Solution===========
- Software Developer Backend Engineer : To oversee Cloud-side development for EMS-Cloud TCP/MQTT/HTTPS communications, API development for data handling in web/apps, and firmware update controls. Utilizing Java/Spring Boot, I focus on internal communication through Kafka and HTTPS, manage ELK stack logging, and assess alternative tech stacks for efficiency. My responsibilities include AWS MSK and OpenSearch monitoring, architecture design for optimal maintenance and performance, and direct partnership collaboration. I aim to deepen my expertise in Kafka, Java, Spring Boot, broaden my backend technology scope, and improve technical communication to effectively convey complex concepts to varied audiences.
- Software Developer DevOps Engineer: basic programming experience in C and C++, along with experience in setting up Linux servers, software development, an understanding of the software development life cycle, Docker, knowledge of CI/CD processes, experience with Jenkins and the Yocto build system, AWS services, and proficiency in JIRA/Confluence. My areas of interest include the overall management of software, from its creation, building, deployment, to management.
- PV Technology Strategy : This role focuses on R&D in solar cell and module technology, involving competitive technology analysis, development of technology roadmaps, adoption and application of new technological trends, understanding technology development trends across the entire value chain, and enhancing technological competitiveness through intellectual property review. The goal is to ensure the company's technology development pace surpasses competitors by swiftly integrating global innovations and trends.
- Software Technology Strategy : This position entails crafting technology roadmaps for PV, focusing on TOPCon and Tandem technologies, utilizing intelligence for strategic R&D planning, and revising IP strategies to pinpoint patenting opportunities and address risks. Duties also include competitive analysis, roadmap development for immediate and future tech applications, and report preparation.
- Legal Managers : Oversee Only Hanwha Qcells related legal compliance and risk management, providing expert legal counsel on corporate law, intellectual property, and regulatory matters, aligning legal strategies with Hanwha QCELLS' business objectives in energy, chemicals, and real estate.
'''

jsonfy_txt2 = '''
===========Output format is given below===========
Evaluate each positions. do not skip evaluating any of position.
1.Relevance score : 0 - 1 as float.
2.Interest score : 0 - 1 as float. 
3.summary : (emoji) 30-words limit.
'''

jsonfy_txt_business = '''
When the user provides an news article, you evaluate the distinct and clear relevance score between hanwha Qcells business and the given news article.
Additionally, Create 3-bullet points(without subject). Each bullet point must have an appropriate emoji in front of sentence.
'''

jsonfy_txt_business2 = '''
===========Output format is given below===========
Evaluate relevance score between hanwha QCELLS business and the article. 
    - Relevance score :  0 - 1 as float.
    - Summary: (emoji)3 bullet points. 20-words limit. 
    
'''

output_template = '''
extract from given text. Not need for key name of each elements in list.
"position names": ["Relevance score", "Interest score", "summary"]
text: ```{text}```
{format_instructions}
'''

output_template2="""
extract from given text. Not need for key name of each elements in list.
["Relevance score", "Summary"]
text: ```{text}```
{format_instructions}
"""

def get_json_parse(texts):
    schema = ResponseSchema(name='output', description="json object, key=Position name, value= [Relevance score, Interest score, summary]")
    output_parser = StructuredOutputParser.from_response_schemas([schema])
    format_instructions = output_parser.get_format_instructions()
    prompt_output = ChatPromptTemplate.from_template(output_template)
    messages = prompt_output.format_messages(text=texts, format_instructions=format_instructions)
    out = llm.chat([ChatMessage(role="user", content=messages[0].content)])
    output = output_parser.parse(out.message.content)
    return output['output']    

def get_json_parse2(texts):
    schema2 = ResponseSchema(name='output', description="list object, value= [Relevance score, summary]")
    output_parser2 = StructuredOutputParser.from_response_schemas([schema2])
    format_instructions2 = output_parser2.get_format_instructions()
    prompt_output2 = ChatPromptTemplate.from_template(output_template2)
    messages = prompt_output2.format_messages(text=texts, format_instructions=format_instructions2)
    out = llm.chat([ChatMessage(role="user", content=messages[0].content)])
    output = output_parser2.parse(out.message.content)
    return output['output']    

def get_raw_news_data():
    df_merge = du.fetch_data(sql = '''select distinct Released_Date , News_Title , News_Contents , News_Author , News_Url , News_tags , News_Image , Updated_Date  from pv_magazine pm where Released_Date = (select max(Released_Date) from pv_magazine)
order by News_Url ''')    
    return df_merge

def get_llm_result(user_message):
    user_message = '===========this is news article as below===========\n' + user_message
    messages = [
        ChatMessage(role="system", content=template),
        ChatMessage(role="system", content=jsonfy_txt),
        ChatMessage(role="system", content=jsonfy_txt2),
        ChatMessage(role="user", content=user_message)]    
    messages2 = [
        ChatMessage(role="system", content=template),
        ChatMessage(role="system", content=jsonfy_txt_business),
        ChatMessage(role="system", content=jsonfy_txt_business2),
        ChatMessage(role="user", content=user_message)]
    response = llm.chat(messages)
    scores = get_json_parse(response.message.content)
    response2 = llm.chat(messages2)
    scores2 = get_json_parse2(response2.message.content)
    return scores, scores2
        
def iterative_prompt(df_merge):
    result_df = pd.DataFrame()
    for i in range(len(df_merge)):
        try:   
            scores, scores2 = get_llm_result(user_message = df_merge.iloc[i].News_Contents)
            print(scores, scores2)
            df = pd.DataFrame.from_dict(scores, orient='index', columns=['Relevance_Score', 'Interesting_Score', 'Summary']).reset_index()
            df['url'] = df_merge.iloc[i].News_Url
            df['title'] = df_merge.iloc[i].News_Title
            df['impact'] = scores2[0]
            df['impact_summary'] = scores2[1]
            result_df = pd.concat([result_df, df], ignore_index=True)
            print('done')
        except Exception as e:
            print(e)
            pass
    return result_df


def html_format(url, title, Relevance_Score , Interesting_Score, impact, impact_summary, summary):
    impact_summary = impact_summary.replace('\n', '<br>')
    summary = summary.replace('\n', '<br>')
    html = f'''
    
        <div align = "left" style="width: 600px; padding: 20px; overflow: auto;font-size: 13px;">
            <a href = {url}><br><span style="font-size: 20px; line-height: 120%">{title}.</span><br></a> 
            <br> releavance score: {Relevance_Score}
            <br> interest score: {Interesting_Score}
            <br> business score: {str(impact)} <br>
            <br>{impact_summary} <br><br> Reason:  {summary}.<br>
        </div>
    
    '''
    
    # html = f'''
    # <td align="center" valign="top" style=" margin: 0; padding: 0; padding-top: 0px;" class="hero">
    #     <div align = "left" style="width: 600px; padding: 20px; overflow: auto;font-size: 13px;">
    #         <a href = {url}><br><span style="font-size: 20px; line-height: 120%">{title}.</span><br></a> 
    #         <br> releavance score: {Relevance_Score}
    #         <br> interest score: {Interesting_Score}
    #         <br> business score: {str(impact)} <br>
    #         <br>{impact_summary} <br><br> Reason:  {summary}.<br>
    #     </div>
    # </td>
    # '''
    html = html
    return html
    
def html_merge(html_element, position_name):
    html = f'''
    <tr>
        <td>
        <hr>
        <div align = "center">
            <a>{position_name}</a>
        </div>
        <table>        
            <tr>
                {html_element}
            </tr>
        </table>					
        </td>
    </tr>
    '''
    return html

def check_asfloat(x):
    try:
        x = float(x)
        return x 
    except:
        return 0.0
        
def parse_to_html(result_df):
    html_merged = ''
    result_df['impact'] = result_df['impact'].apply(lambda x:check_asfloat(x))
    result_df_ = result_df[result_df['Interesting_Score'] != '']
    for position_name in result_df_['position'].unique():
        result_df_ = result_df[(result_df['Relevance_Score']>=0.7) & (result_df['Interesting_Score']>=0.7)]
        result_df_ = result_df_[result_df_['position'] == position_name]
        if len(result_df_) > 0:
            result_df_['html'] = result_df_.apply(lambda x: html_format(x['url'], x['title'], x['Relevance_Score'], x['Interesting_Score'], x['impact'], x['impact_summary'], x['Summary']), axis = 1)
            result_df_ = result_df_.sort_values(by = 'impact', ascending = False).head(3)
            htmls = result_df_['html'].values
            html_merged = html_merged + html_merge('\n'.join(htmls[:3]), position_name)
    return html_merged
    
def save_html(html_merged):
    with open("../tmp/base_template.html", "r", encoding='utf-8') as f:
        text = f.read()
    text = text.replace('datainput', html_merged)
    with open("../data/output/news_template.html", "w", encoding='utf-8') as f:
        f.write(text)

def news_main():
    df_merge = get_raw_news_data()
    result_df = iterative_prompt(df_merge)
    result_df = result_df.rename(columns = {'index' : 'position'})
    du.insert_pd_tosql('news_llm_output', result_df)
    # result_df = du.fetch_data('''select * from pv_magazine pm where Released_Date = (select max(Released_Date) from pv_magazine)''')
    print(result_df)
    html_merged = parse_to_html(result_df)
    save_html(html_merged)

if __name__ == '__main__':
    news_main()
