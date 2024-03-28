from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import os
import glob
import multiprocessing
import chromedriver_autoinstaller
chromedriver_autoinstaller.install()

class global_obj(object):
    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("detach", True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

urls = ['https://www.bisnow.com/atlanta/news/industrial/qcells-inks-major-warehouse-lease-near-cartersville-plant-123268',
 'https://www.reuters.com/sustainability/climate-energy/microsoft-qcells-strike-massive-supply-deal-us-made-solar-panels-2024-01-08/',
 'https://www.solarpowerworldonline.com/2024/02/qcells-inks-module-recycling-partnership-with-solarcycle/',
 'https://www.ajc.com/news/georgia-solar-will-qcells-new-plant-usher-in-an-american-made-solar-revival/OCKGUCKVNZAERMSJZVJVSA5NGU/',
 'https://www.orrick.com/en/News/2024/01/Microsoft-and-Qcells-Reach-Historic-12GW-Renewable-Energy-Agreement',
 'https://www.prnewswire.com/news-releases/qcells-announces-expansion-of-strategic-alliance-with-microsoft-302027749.html',
 'https://www.commercialsearch.com/news/qcells-signs-843-ksf-georgia-lease/',
 'https://www.pv-tech.org/qcells-inks-recycling-agreement-with-solarcycle/',
 'https://www.pv-magazine.com/2024/02/13/qcells-solarcycle-aim-to-jointly-recover-95-of-solar-panel-value/',
 'https://fortune.com/2023/10/18/qcells-georgia-solar-panel-factory/']

result_queue = multiprocessing.Queue()
driver = webdriver.Chrome(options=global_obj.chrome_options)
driver.delete_all_cookies()
driver.set_page_load_timeout(10)
driver.execute_script("window.open('');")
driver.execute_script("window.open('');")
driver.execute_script("window.open('');")
driver.execute_script("window.open('');")
t1 = time.time()

def get_string_from_news(idx, url, driver):    
    try:
        driver.switch_to.window(driver.window_handles[idx])
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        anchor_elements = soup.find_all('p')
        result_string = '\n'.join([i.text.strip() for i in anchor_elements])  
        driver.quit()
        result_queue.put(Document(text=result_string[:20000], metadata = {'url' : url}))
        print('\n', idx, url, driver, time.time() - t1)
    except:
        driver.quit()
        pass

if __name__ == '__main__':
    processes = []
    for idx, url in enumerate(urls[:5]):
        p = multiprocessing.Process(target=get_string_from_news, kwargs={'idx': idx, 'url': url, 'driver':driver})
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    p.close()
