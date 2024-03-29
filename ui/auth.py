import streamlit as st
from httpx_oauth.clients.microsoft import MicrosoftGraphOAuth2
import asyncio
import nest_asyncio
import requests

nest_asyncio.apply()

CLIENT_ID = 'f17632ac-7fc4-4525-a157-518f7cbcdc8d'
CLIENT_SECRET = '3k48Q~HELekDJTlvz_vAAVXSSi-JoJshp~cPPc7z'
REDIRECT_URI = 'https://qcells-us-rag.westus2.cloudapp.azure.com:442/'
TENANT_ID = '0f7b4e1c-344e-4923-aaf0-6fca9e6700c8'

async def get_authorization_url(client: MicrosoftGraphOAuth2, redirect_uri: str):
    authorization_url = await client.get_authorization_url(redirect_uri, scope=["f17632ac-7fc4-4525-a157-518f7cbcdc8d/.default"])
    return authorization_url
    
def get_login_str():
    client: MicrosoftGraphOAuth2 = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
    authorization_url = asyncio.run(get_authorization_url(client, REDIRECT_URI))
    return authorization_url
    
async def get_email(client: MicrosoftGraphOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email


async def get_access_token(client: MicrosoftGraphOAuth2, redirect_uri: str, code: str):
    token = await client.get_access_token(code, redirect_uri)
    return token
    
def display_user():
    try:
        client: MicrosoftGraphOAuth2 = MicrosoftGraphOAuth2(CLIENT_ID, CLIENT_SECRET)
        code = st.query_params.get_all('code')
        token = asyncio.run(get_access_token(client, REDIRECT_URI, code))
        if code:
            return True
        else:
            return False
    except:
        pass
st.link_button("Sign In", get_login_str())
    