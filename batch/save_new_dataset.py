import glob
import pandas as pd
import os 
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, PandasQueryEngine
import openai
from datetime import datetime
import nest_asyncio
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import Document
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from chromadb.config import Settings
import sys
sys.path.append('../utils')
from db_utils import * 
du = DB_Utils()

nest_asyncio.apply()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'
os.environ['ALLOW_RESET']='TRUE'
embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")
llm = AzureChatOpenAI(
    temperature = 0
    , deployment_name="test_gpt"
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding,
)
set_global_service_context(service_context)

class save_node_to_vectordb():
    def __init__(self):
        db = chromadb.HttpClient(host='localhost', port=8000, settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                                                                                chroma_client_auth_credentials="qcells:qcells"))
        self.chroma_collection = db.get_collection("pv_magazine_sentence_split")

    def remove_duplicates(self):
        print('delete duplicates')
        sql= '''
        WITH CTE AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY News_Url ORDER BY (SELECT 0)) AS rn
            FROM pv_magazine
        )
        DELETE FROM CTE
        WHERE rn > 1;
        '''
        du.sql_execute_commit(sql)
    
    def formatting_news(self, url, news_national_type, tag, author, updated_date, title, contents):
        news_article = f'''
            1. news url: {url}
            3. news nationality: {news_national_type}
            4. news tag: {tag}
            5. news author name: {author}
            6. news update date: {updated_date}
            7. news title: {title}
            8. news contents: 
            {contents}
            '''
        return news_article 
    
    def check_url_duplicates(self, url):
        if url in self.urls:
            return True
        else:
            return False
    
    def make_doc(self, df):
        if self.check_url_duplicates(df.News_Url)==False:
            doc = Document(text=df.format_txt,
                            metadata={
                                        "release_date_year": int(df.Released_Date.year),
                                        "release_date_month": int(df.Released_Date.month),
                                         "author": df.News_Author,
                                         "national": df.News_Nationality,
                                         "tag": df.News_tags,
                                         "url" : df.News_Url
                                       }
                          )
            return doc
    
    def get_url_saved_url(self):
        chroma_saved = self.chroma_collection.get(where={"release_date_year": {"$eq": datetime.now().year}})
        urls = [i['url'] for i in chroma_saved['metadatas']]
        urls = list(set(urls))
        return urls
    
    def save_new_dataset(self):
        print('save to vectordb')
        self.df_merge['format_txt'] = self.df_merge.apply(lambda x:self.formatting_news(x['News_Url'], x['News_Nationality'], x['News_tags'], x['News_Author'], x['Released_Date'], x['News_Title'], x['News_Contents']), axis = 1)
        documents = [self.make_doc(self.df_merge.iloc[i]) for i in range(len(self.df_merge)) if self.make_doc(self.df_merge.iloc[i]) is not None]
        splitter = SentenceSplitter(chunk_size=512,chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store= vector_store)
        service_context = ServiceContext.from_defaults(embed_model=embedding)
        VectorStoreIndex(nodes=nodes, storage_context=storage_context, service_context=service_context, show_progress=True)

    def save_main(self):
        self.remove_duplicates()
        self.df_merge = du.fetch_data(sql = '''select * from pv_magazine pm where Released_Date = (select max(Released_Date) from pv_magazine)''')    
        self.urls = self.get_url_saved_url()
        self.save_new_dataset()
        
if __name__ == '__main__':
    s = save_node_to_vectordb()
    s.save_main()