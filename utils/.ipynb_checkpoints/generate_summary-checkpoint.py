from llama_index.core import SummaryIndex
from llama_index.core import PromptTemplate
from pydantic import Field, BaseModel
from llama_index.core import SummaryIndex, get_response_synthesizer, VectorStoreIndex, Document, KeywordTableIndex,ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram

class summary(BaseModel):
    """
        Restructure answers.    
        Return:
            youtube_play_time: hh:mm:ss - hh:mm:ss
            Summary: 
    """
    play_time: str
    Summary: str

class Text_summary(BaseModel):
    """
        Restructure answers. If there is nothing to summarize, Return blank.
        Return:
            Summary: 
    """
    Summary: str
    
class Youtube_answer(BaseModel):
    """
        Restructure answers.    
        Return:
            emoji: 
            play_time: hh:mm:ss
            Summary: 
    """
    emoji: str
    play_time: str
    Summary: str

class Text_answer(BaseModel):
    """
        Restructure answers.    
        Return:
            emoji: 
            Summary: 
    """
    emoji: str
    Summary: str

class map_reduced_summary():
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding
        self.service_context = ServiceContext.from_defaults(llm=llm,embed_model=embedding,)
        self.router_prompt0 = PromptTemplate(
            "The following is a compilation of summaries:"
            "\n---------------------\n{context_list}\n---------------------\n"
            "Based on these, please create an summary. Also, add an emoji that suit the content.\n"
            "Answer:")        
            
    def create_youtube_summary(self, text):
        if (len(text) / 5) > 2048:
            chunk_size = 2048
        elif (len(text) / 5) > 1024:
            chunk_size = 1024   
        else:
            chunk_size = 512
        
        self.splitter = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=int(chunk_size/10))        
        map_template = PromptTemplate(
            "The following is a partial content from the document.\n"
            "{context_str}\n"
            "Please provide summary in English within 50-words limited. Add play start time and play end time.\n"
            "<start time - end time> Summary: "
        )
        self.response_synthesizer = get_response_synthesizer(text_qa_template=map_template, llm = self.llm, response_mode="accumulate", output_cls = summary)
        documents = Document(text=text)
        index = SummaryIndex.from_documents([documents], service_context = self.service_context, transformations=[self.splitter])
        query_engine = index.as_query_engine(response_synthesizer=self.response_synthesizer, use_async=True, verbose = True)
        response = query_engine.query("Summarize the provided text.", )    
        self.split_res = response.response.split('---------------------')
        program = OpenAIPydanticProgram.from_defaults(output_cls=Youtube_answer,prompt=self.router_prompt0,llm=self.llm)
        map_res = []
        for i in self.split_res:
            output = program(context_list=i) 
            output = output.play_time + ':\n' + output.emoji + output.Summary
            map_res.append(output)
        final_result = ('\n\n').join(map_res)
        return final_result

    def create_document_summary(self, text):
        if (len(text) / 5) > 2048:
            chunk_size = 2800
        elif (len(text) / 5) > 1024:
            chunk_size = 1400   
        else:
            chunk_size = 1024
            
        self.splitter = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=(chunk_size/10))        
        map_template = PromptTemplate(
            "The following is a partial content from the document.\n"
            "{context_str}\n"
            "Please provide summary with key point in English within 50 characters or fewer.\n"
            "Summary: ")
        self.response_synthesizer = get_response_synthesizer(text_qa_template=map_template, llm = self.llm, response_mode="accumulate", output_cls = Text_summary)
        
        documents = Document(text=text)
        index = SummaryIndex.from_documents([documents], service_context = self.service_context, transformations=[self.splitter])
        query_engine = index.as_query_engine(response_synthesizer=self.response_synthesizer, use_async=True, verbose = True)
        response = query_engine.query("Summarize the provided text.")    
        self.split_res = response.response.split('---------------------')
        program = OpenAIPydanticProgram.from_defaults(output_cls=Text_answer,prompt=self.router_prompt0,llm=self.llm)
        map_res = []
        for i in self.split_res:
            output = program(context_list=i) 
            output = output.emoji + output.Summary
            map_res.append(output)
        final_result = ('\n\n').join(map_res)
        return final_result


