import requests
import streamlit as st
from chatgpt_utils import set_llm, set_rag, set_llm4, set_embedding, get_tutorial_tech_sensing, get_tutorial_gpt, get_anno_tech_sensing, get_session_init, get_login_str, get_authorization_url, CLIENT_ID2, REDIRECT_URI2, CLIENT_SECRET2, TENANT_ID2, get_login_str_de
import asyncio
from httpx_oauth.clients.microsoft import MicrosoftGraphOAuth2

def get_access_token(CLIENT_ID, REDIRECT_URI, CLIENT_SECRET, TENANT_ID):
    payload = {
        'client_id': CLIENT_ID,
        'scope': 'User.Read offline_access',
        'code': st.query_params.get_all('code'),
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code',
        'client_secret': CLIENT_SECRET
    }        
    response = requests.post(
                            'https://login.microsoftonline.com/{}/oauth2/v2.0/token'.format(TENANT_ID)
                             , headers={'Content-Type': 'application/x-www-form-urlencoded'}
                             , data=payload)
    return response

with st.spinner('login...'):
    response = get_access_token(CLIENT_ID2, REDIRECT_URI2, CLIENT_SECRET2, TENANT_ID2)
    # st.write(response.json())
    if 'error' not in response.json():
        st.session_state.refresh_token = response.json()['refresh_token']
        responses = requests.get(
                                'https://graph.microsoft.com/v1.0/me',
                                 headers={'Authorization': 'Bearer ' + response.json()['access_token']})
        
        # st.write(responses.json())
        st.session_state.access_mail = responses.json()['mail']
        st.session_state.is_signed_in = True
        st.switch_page("main.py")
    else:
        authorization_url = get_login_str_de()
        # st.write(authorization_url)
        st.markdown(f"""
            <meta http-equiv="refresh" content="0; URL={authorization_url}">
        """, unsafe_allow_html=True)        
        # response = get_access_token(CLIENT_ID2, REDIRECT_URI2, CLIENT_SECRET2, TENANT_ID2)
        # st.write(response.json())
        