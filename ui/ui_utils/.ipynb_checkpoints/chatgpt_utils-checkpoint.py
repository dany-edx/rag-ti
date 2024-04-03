from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

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
    