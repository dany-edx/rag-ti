
import os
import datetime
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.ai.translation.document import DocumentTranslationClient
from azure.storage.blob import BlobServiceClient, BlobClient, generate_container_sas


class SampleTranslationWithAzureBlob:
    def __init__(self):
        self.endpoint = 'https://rag-translator.cognitiveservices.azure.com/'
        self.key = '26d9b4d5fa8141d5a4abc27bfba5875d'
        self.storage_endpoint = 'https://qcellsragblobs.blob.core.windows.net'
        self.storage_account_name = 'qcellsragblobs'
        self.storage_key = 'sRYFLWXylmTSpS/jOJtRy2g+6HR8hv5620FTPU3ztfm7gCMG93K/aX9M/B5p1Uqk1v+o+osLjTRz+ASt3Y+nrQ=='
        self.storage_source_container_name = 'target-container'
        self.storage_target_container_name = 'translated-container'
        self.translated_dir_path = '../tmp/translated/'
        # self.document_name = 'ATP_결과데이터 분석.pptx'
        # self.document_name = '230725_연구소_2023년 상반기 태양광 산업동향_암호화해제.pdf'

    def sample_translation_with_azure_blob(self, document_name, to_lang = 'en'):
        self.document_name = document_name
        translation_client = DocumentTranslationClient(self.endpoint, AzureKeyCredential(self.key))
        blob_service_client = BlobServiceClient(self.storage_endpoint,credential=self.storage_key)
        source_container = self.create_container(blob_service_client,container_name=self.storage_source_container_name)
        target_container = self.create_container(blob_service_client,container_name=self.storage_target_container_name)
                    
        for blob in source_container.list_blobs():
            print(blob.name, self.document_name)
            if blob.name == self.document_name:   
                source_container.delete_blobs(self.document_name)
                
        for blob in target_container.list_blobs():
            print(blob.name, self.document_name)
            if blob.name == self.document_name:   
                target_container.delete_blobs(self.document_name)
        
        with open(self.translated_dir_path + self.document_name, "rb") as doc:
            source_container.upload_blob(self.document_name, doc)
        print(f"Uploaded document {self.document_name} to storage container {source_container.container_name}")

        source_container_sas_url = self.generate_sas_url(source_container, permissions="rl")
        target_container_sas_url = self.generate_sas_url(target_container, permissions="wl")

        source_container_sas_url = source_container_sas_url.replace('%3A', ':')
        target_container_sas_url = target_container_sas_url.replace('%3A', ':')
        
        poller = translation_client.begin_translation(source_container_sas_url, target_container_sas_url, to_lang)
        print(f"Created translation operation with ID: {poller.id}")
        print("Waiting until translation completes...")
        result = poller.result()
        print(f"Status: {poller.status()}")
        print("\nDocument results:")
        for document in result:
            print(f"Document ID: {document.id}")
            print(f"Document status: {document.status}")
            if document.status == "Succeeded":
                print(f"Source document location: {document.source_document_url}")
                print(f"Translated document location: {document.translated_document_url}")
                print(f"Translated to language: {document.translated_to}\n")
                blob_client = BlobClient.from_blob_url(document.translated_document_url, credential=self.storage_key)
                with open(self.translated_dir_path +  self.document_name, "wb") as my_blob:
                    download_stream = blob_client.download_blob()
                    my_blob.write(download_stream.readall())
                print("Downloaded {} locally".format("translated_"+self.document_name))
            else:
                print("\nThere was a problem translating your document.")
                print(f"Document Error Code: {document.error.code}, Message: {document.error.message}\n")
        self.download_blob_to_file()
        for blob in source_container.list_blobs():
            if blob.name == self.document_name:   
                source_container.delete_blobs(self.document_name)
        for blob in target_container.list_blobs():
            if blob.name == self.document_name:   
                target_container.delete_blobs(self.document_name)

    def create_container(self, blob_service_client, container_name):
        try:
            container_client = blob_service_client.create_container(container_name)
            print(f"Creating container: {container_name}")
        except ResourceExistsError:
            print(f"The container with name {container_name} already exists")
            container_client = blob_service_client.get_container_client(container=container_name)
        return container_client

    def generate_sas_url(self, container, permissions):
        sas_token = generate_container_sas(
            account_name=self.storage_account_name,
            container_name=container.container_name,
            account_key=self.storage_key,
            permission=permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        )
        container_sas_url = self.storage_endpoint + '/' +container.container_name + "?" + sas_token
        print(f"Generating {container.container_name} SAS URL")
        return container_sas_url

    def download_blob_to_file(self):
        blob_service_client = BlobServiceClient(self.storage_endpoint,credential=self.storage_key)
        blob_client = blob_service_client.get_blob_client(container=self.storage_target_container_name, blob=self.document_name)
        with open(self.translated_dir_path + 'translated_' + self.document_name, "wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())

if __name__ == '__main__':
    sample = SampleTranslationWithAzureBlob()
    poller = sample.sample_translation_with_azure_blob('230725_연구소_2023년 상반기 태양광 산업동향_암호화해제.pdf')