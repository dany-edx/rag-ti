from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core import set_global_service_context,  ServiceContext, Document,get_response_synthesizer
import sys
sys.path.append('../utils')
from generate_summary import * 
from pytube import Channel,YouTube #키워드 -> url
import os 
import numpy as np
from datetime import datetime, timedelta
import nest_asyncio
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import nest_asyncio
import pandas as pd
import glob

nest_asyncio.apply()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'
os.environ['ALLOW_RESET']='TRUE'

youtube_prompt = '''
Structure the 5-6 content into a table of contents and a clear and academical delineation that highlights its unique context and narrative. 
For each entry in the table of contents, include a business insight or achievements(how much) using precise technical terminology in order to learn new technologies.

===========Output format is given below===========
* Keywords: #keywords
1. {table of contents} (emoji)20-words limit.
    - {description}  20-words limit.
'''


llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="test_gpt",
    temperature = 0,
    api_key="70d67d8dd17f436b9c1b4e38d2558d50",
    azure_endpoint="https://qcells-us-test-openai.openai.azure.com/",
    api_version="2023-07-01-preview",
)

embedding = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="embedding_model",
    api_key="70d67d8dd17f436b9c1b4e38d2558d50",
    azure_endpoint="https://qcells-us-test-openai.openai.azure.com/",
 api_version="2023-07-01-preview",
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding,
)
set_global_service_context(service_context)

class get_youtube_data():
    def __init__(self):
        self.channels = {'PV' : ['photovoltaicsint', 'TaiyangNewsAllAboutSolar', 'TrinaSolarPV', 'solarenergyengineering1162', 'sinovoltaics', 'atainsightschannel234', 'globaligp', 'globalenergycommunity9094', 'Solarable','SolarSurge'   ],
                         'ESS' : ['DerKanal', 'cleanpowerhour123', 'SolarUnitedNeighbors','BriggsandStrattonEnergy','DigitalEU'],
                         'VPP' : ['EEXTV1', 'AutoGridSystems', 'SolaxPowerGlobal', 'PacificSunTech'],
                         'Software' : ['infoq'],
                         'Cloud' : ['AWSEventsChannel', 'BestDotNetTraining'],
                         'LLM' : ['LlamaIndex','element451crm']
                         }
        
    def get_timeline(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '{}:{}:{}'.format(int(h), format(int(m), '02'), np.round(s,1))
    
    def get_metadata(self,url):
        yt = YouTube(url)
        video_info = {
            "url":url,
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown",
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date
            else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
        }
        return video_info

    def get_data(self, url):
        self.url = url
        meta_data = self.get_metadata(url) 
        data = YouTubeTranscriptApi.get_transcript(self.url.split('v=')[-1])
        documents = []
        for i in data:
            i['div'] = int(i['start'] / 300)  
            
        for idx, d in enumerate(range(0, data[-1]['div']+1)):
            texts = []
            k = [i for i in data if i['div'] == d]
            start_time_min = min([i['start'] for i in k])
            start_time_max = max([i['start'] for i in k])
            text_div = [i['text'] for i in k]
            texts.append('[youtube play time range] {} - {}\n'.format(self.get_timeline(start_time_min), self.get_timeline(start_time_max)) + ' '.join(text_div))
            texts = '\n'.join(texts)
            documents.append(Document(text=texts, metadata= {'title': meta_data['title']  ,'page_id': idx, 'play_time_from' : self.get_timeline(start_time_min),'play_time_to': self.get_timeline(start_time_max)}))
        return documents, meta_data['thumbnail_url']
    
    def channel_exploration(self):
        analyzed_data = {}
        for channel_key in self.channels.keys():
            print(channel_key)
            for ch in self.channels[channel_key]:
                channel_vs = Channel(ch)
                recent_videos = []
                for i, c in enumerate(channel_vs[:5]):
                    # loader = YoutubeLoader.from_youtube_url(c, add_video_info=True)
                    try:
                        metainfo = self.get_metadata(c) 
                        if len(metainfo) > 0:
                            print(i, metainfo['title'], metainfo['author'])
                            if datetime.strptime(metainfo['publish_date'],'%Y-%m-%d %H:%M:%S') >= datetime.strptime(datetime.strftime(datetime.now(),'%Y%m%d'), '%Y%m%d') - timedelta(days = 5):
                                recent_videos.append(c) 
                    except:
                        pass
                analyzed_data[channel_key] = recent_videos
        return analyzed_data

    def analyze_documents(self, analyzed_data):
        result_df = pd.DataFrame()
        for key in analyzed_data.keys():
            print(key)
            for url in analyzed_data[key]:
                try:
                    print(url)
                    documents, thumbnail_url = self.get_data(url = url)
                    if len(documents) > 2:
                        mrs = map_reduced_summary(llm = llm)
                        llm_result = mrs.create_youtube_summary(text = '\n'.join([i.text for i in documents]))
                        df = pd.DataFrame([key, url, llm_result, thumbnail_url]).T
                        result_df = pd.concat([result_df, df])
                except:
                    pass
        print(result_df)
        result_df.columns = ['domain', 'url', 'summary', 'thumbnail_url']
        return result_df

    def llm_engine(self, documents):
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)
        doc_summary_index = DocumentSummaryIndex.from_documents(documents,service_context=service_context,response_synthesizer=response_synthesizer,)
        query_engine = doc_summary_index.as_query_engine(response_mode="refine", use_async=True)
        response = query_engine.query(youtube_prompt)
        return response


def html_format(youtube_url, thumbnail_url, llm_result):
    llm_result = llm_result.replace('\n','<br>')
    component_html = f'''   
    <tr>
        <td style="padding: 20px; text-align: left;">
            <a style="display: block;" href="https://www.youtube.com/watch?v={youtube_url}">
                <img src="{thumbnail_url}" width="100%" style="display: block; margin: 0 auto;" />
            </a>
        </td>
    </tr>
    
    <tr>
        <td style="padding: 20px; text-align: left;">
            <div style="word-wrap: break-word; font-size: 12px;">
            {llm_result}
            </div>
        </td>
    </tr>
    '''
    return component_html
       
def parse_to_html(result_df):
    result_df['url_id'] = result_df['url'].apply(lambda x : x.split('v=')[-1])
    result_df['html'] = result_df.apply(lambda x:html_format(x['url_id'], x['thumbnail_url'], x['summary']), axis = 1)
    html_merged = []
    for domain_name in result_df.domain.unique():
        domain_merged = ''.join(result_df[result_df['domain']==domain_name]['html'].values)
        div_info = f'''

        <tr>
            <td style="padding: 5px;">
                <hr>
                <a style="color: #000000; text-decoration: none; display: block; font-size: 12px;">{domain_name}</a>
            </td>
        </tr>
                        
        <tr>
            {domain_merged}
        </tr>
                        '''
        html_merged.append(div_info)

    return html_merged
    
def save_html(html_merged):
    with open("../data/output/news_template.html", "r", encoding='utf-8') as f:
        text = f.read()
        
    text = text.replace('youtubeinput', '\n'.join(html_merged))
    
    with open("../data/output/send_html.html", "w", encoding='utf-8') as f:
        f.write(text)

def youtube_main():
    gy = get_youtube_data()
    analyzed_data = gy.channel_exploration()
    result_df = gy.analyze_documents(analyzed_data)
    if len(result_df) > 0:
        result_df.to_csv('../data/output/youtube_result.csv')
    else:
        result_df = pd.read_csv('../data/output/youtube_result.csv')
    html_merged = parse_to_html(result_df)
    save_html(html_merged)

if __name__ == '__main__':
    youtube_main()

# search_datetime=datetime.strftime(datetime.now()- timedelta(days = 2),'%Y%m%d')
    #vectordb insert
    #thumbnail url
    