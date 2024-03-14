import threading
import time
from datetime import datetime, timedelta
from emailing import *
from news_prompt_v2 import * 
from pv_magazine import pv_main
from save_new_dataset import * 
from youtube_prompt import * 

class batch_schedule():
    
    def tasks(self):
        pv_main()
        news_main()
        youtube_main()
        email_sender()
        s = save_node_to_vectordb()
        s.save_main()
        del s
    
        
    def scheduled_task(self):
        while True:
            current_time = datetime.now().time()
            target_time = datetime.strptime('21:11:00', '%H:%M:%S').time()
    
            if current_time >= target_time:
                print("작업을 실행합니다.")
                self.tasks()
                
                target_time = (datetime.now() + timedelta(days=1)).replace(hour=13, minute=0, second=0)
                sleep_time = (target_time - datetime.now()).total_seconds()
                time.sleep(sleep_time)
            else:
                time.sleep(1)

# 쓰레드 생성 및 시작
bs = batch_schedule()
thread = threading.Thread(target=bs.scheduled_task)
thread.start()

if __name__ == '__main__':
    pv_main()
    news_main()
    youtube_main()
    email_sender()
    s = save_node_to_vectordb()
    s.save_main()
