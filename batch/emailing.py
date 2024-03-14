from O365 import Account, FileSystemTokenBackend
from datetime import datetime, timedelta

def email_sender():
    graph_token_path = '../tmp/o365_token.txt'
    tenant_id = '133df886-efe0-411c-a7af-73e5094bbe21'
    app_id = "e87f80a9-65e5-4d54-97b0-82175ada5c60"
    my_client_secret = "fOW8Q~GOH3hqT1oaJpN~9HUeUJs~A6JzY_kWbcm."
    # dev_address = ['dany.shin@hanwha.com']
    dev_address = ['dany.shin@hanwha.com','hwkwak@qcells.com', 'jaehong.koo@qcells.com', 'max.morrison@qcells.com', 'kyuhoon.lim@hanwha.com']    
    account = Account(credentials = (app_id, my_client_secret)
                    , scopes = ['basic', 'message_all']
                    , token_backend= FileSystemTokenBackend(token_filename=graph_token_path)
                    , tenant_id=tenant_id)
    
    f = open("../data/output/send_html.html", 'r')
    test_mail = f.read()
    f.close()
    
    m = account.new_message() #create email instance
    m.to.add(dev_address) #destination address
    m.subject = '[LLM] Today information Youtube & NEWS'
    m.body = test_mail 
    m.send()

if __name__ == '__main__':
    email_sender()