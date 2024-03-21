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
import sys
sys.path.append('../utils')
from db_utils import DB_Utils 
chromedriver_autoinstaller.install()
du = DB_Utils()
class global_obj(object):
    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("detach", True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}

class get_pv_magazine_global():
    def article_preview_global(self):
        while True:
            enter_articles = self.driver.find_elements(By.CLASS_NAME, 'article-preview')
            for art_idx, _ in enumerate(enter_articles):
                try:
                    self.selected_article_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'image-wrap').find_element(By.TAG_NAME, 'a').get_attribute('href')
                    self.selected_article_img_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'image-wrap').find_element(By.TAG_NAME, 'a').find_element(By.TAG_NAME, 'img').get_attribute('src')
                except:
                    self.selected_article_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'entry-title').find_element(By.TAG_NAME, 'a').get_attribute('href')
                    pass
                self.driver.get(self.selected_article_url)
                self.get_contents()
                time.sleep(1)
            is_nav = len(self.driver.find_element(By.CLASS_NAME, 'pagination').find_elements(By.TAG_NAME, 'nav'))
            if is_nav>0:
                self.url_next = self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div[3]/div[10]/div/div[1]/nav/div/nav/div/a[2]').get_attribute('href')
                if 'page' in self.url_next:
                    self.driver.get(self.url_next)
                else:
                    break
            else:
                break
            
class get_pv_magazine(get_pv_magazine_global, DB_Utils):
    def __init__(self, base_url, dir_name, type = 'national'):
        self.driver = webdriver.Chrome( options=global_obj.chrome_options)
        self.driver.implicitly_wait(0)
        self.dataset = []
        self.base_url = base_url
        self.dir_name = dir_name
        self.type = type
        self.start_date = self.get_saved_max_date()
        self.end_date = datetime.now()
        self.dates_list = self.get_dates()

    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    def get_saved_max_date(self):
        updated_df = du.fetch_data(sql = f'''select 
                                            CONVERT(CHAR(8), Convert(Date, max(Released_Date)), 112)
                                            from pv_magazine pm 
                                            where News_Nationality = '{self.dir_name}'
                                            ''')
        
        # max_saved_date = updated_df
        max_saved_date = updated_df.values[0][0]
        max_saved_date = datetime.strptime(max_saved_date,'%Y%m%d')
        
        return max_saved_date

    def get_dates(self):
        dates_list = []
        current_date = self.start_date
        while current_date <= self.end_date:
            formatted_date = current_date.strftime("%Y/%m/%d")
            dates_list.append(formatted_date)
            current_date += timedelta(days=1)
        return dates_list

    def check_folder_to_csv(self, file_path):
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        # self.df.to_csv(file_path + '/result.csv', index = False, encoding='utf-8-sig')

    def access_main_page(self):
        for self.date in self.dates_list:
            try:
                self.dataset = []
                self.url = self.base_url + "/{}/".format(self.date) #날짜별로 다운로드!!!!!
                print(self.url)
                self.driver.get(self.url)       
                time.sleep(3) 
                if len(self.driver.find_elements(By.CLASS_NAME, 'article-preview')) == 0:
                    continue
                else:
                    if self.type == 'national':
                        self.article_preview()
                    else:
                        if self.url == self.driver.current_url:
                            self.article_preview_global()
                    # self.df = pd.DataFrame(self.dataset, columns = ['url','tag','author','updated_date','title', 'contents', 'img_url', 'national'])
                    self.check_folder_to_csv(file_path = '../data/pvmagazine/{}/{}'.format(self.dir_name, self.date.replace('/','')))
                    self.df = pd.DataFrame(self.dataset, columns = ['News_Url','News_tags','News_author','Released_Date','News_Title', 'News_Contents', 'News_Image', 'News_Nationality'])
                    self.df.to_csv('../data/pvmagazine/{}/{}/result.csv'.format(self.dir_name, self.date.replace('/','')))                
                    self.insert_pd_tosql(tablename = 'pv_magazine', df = self.df)
            except exception as e:
                pass
            
    def article_preview(self):
        while True:
            enter_articles = self.driver.find_elements(By.CLASS_NAME, 'article-preview')
            for art_idx, _ in enumerate(enter_articles):
                try:
                    self.selected_article_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'image-wrap').find_element(By.TAG_NAME, 'a').get_attribute('href')
                    self.selected_article_img_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'image-wrap').find_element(By.TAG_NAME, 'a').find_element(By.TAG_NAME, 'img').get_attribute('src')
                except:
                    self.selected_article_url = self.driver.find_elements(By.CLASS_NAME, 'article-preview')[art_idx].find_element(By.CLASS_NAME, 'entry-title').find_element(By.TAG_NAME, 'a').get_attribute('href')
                    pass
                self.driver.get(self.selected_article_url)
                self.get_contents()
                time.sleep(1)
            is_nav = len(self.driver.find_element(By.CLASS_NAME, 'pagination').find_elements(By.TAG_NAME, 'nav'))
            if is_nav>0:
                self.url_next = self.driver.find_element(By.XPATH, '/html/body/div[1]/div/div[3]/div[10]/div/div[1]/nav/div/nav/div/a[2]').get_attribute('href')
                if 'page' in self.url_next:
                    self.driver.get(self.url_next)
                else:
                    break
            else:
                break
            
    def get_contents(self):
        totalcontents = self.driver.find_element(By.CLASS_NAME, 'singular-inner')
        self.selected_article_title = totalcontents.find_element(By.CLASS_NAME, 'entry-title').text
        try:
            self.summary = totalcontents.find_element(By.CLASS_NAME, 'entry-byline').find_element(By.CLASS_NAME, 'article-lead-text').text
        except:
            self.summary = ''
            pass
        self.contents = [i.text for i in totalcontents.find_element(By.CLASS_NAME, 'entry-content').find_elements(By.TAG_NAME, 'p')]
        self.selected_article_contents = ('\n').join(self.contents)
        self.selected_article_date = totalcontents.find_element(By.CLASS_NAME, 'entry-published.updated').text
        try:
            self.selected_article_author = totalcontents.find_element(By.CLASS_NAME, 'author.url.fn').text
        except:
            self.selected_article_author = ''
            pass
        self.selected_article_tag = ','.join([i.text for i in totalcontents.find_elements(By.CLASS_NAME, 'nav-link')])
        print(self.date)
        print('=' * 30, 'NEWS' ,'=' * 30,'\n')
        print(self.selected_article_url)
        print(self.selected_article_title)
        
        self.dataset.append([self.selected_article_url, self.selected_article_tag, self.selected_article_author, self.date.replace('/','-'), self.selected_article_title, self.selected_article_contents, self.selected_article_img_url, self.dir_name])
        self.driver.back()

    def main(self):
        try:
            self.access_main_page()
        except Exception as error:
            print(error)
        finally:
            self.driver.close()
            pass

def process_pv_magazine(base_url, dir_name, type = 'national'):
    pv = get_pv_magazine(base_url=base_url, dir_name=dir_name, type = type)
    pv.access_main_page()


def pv_main():
    instances = [
        {'base_url': 'https://www.pv-magazine-australia.com', 'dir_name': 'au'},
        {'base_url': 'https://www.pv-magazine-brasil.com', 'dir_name': 'brasil'},
        {'base_url': 'https://pv-magazine.fr', 'dir_name': 'france'},
        {'base_url': 'https://www.pv-magazine.de', 'dir_name': 'germany'},
        {'base_url': 'https://www.pv-magazine-india.com', 'dir_name': 'india'},
        {'base_url': 'https://www.pv-magazine-china.com', 'dir_name': 'china'},
        {'base_url': 'https://pv-magazine.it', 'dir_name': 'italy'},
        {'base_url': 'https://www.pv-magazine-usa.com', 'dir_name': 'usa'},
        {'base_url': 'https://www.pv-magazine.com', 'dir_name': 'global', 'type':'global'},
    ]

    processes = []
    for instance in instances:
        p = multiprocessing.Process(target=process_pv_magazine, kwargs=instance)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    updated_df = du.fetch_data(sql = f'''select 
                                        CONVERT(CHAR(8), Convert(Date, max(Updated_Date)), 112)
                                        from pv_magazine pm 
                                        ''')
    max_saved_date = updated_df.values[0][0]
    csv_list = glob.glob('../data/pvmagazine/**/{}/result.csv'.format(max_saved_date))
    dfs = pd.DataFrame()
    for i in csv_list:
        df = pd.read_csv(i)
        dfs = pd.concat([dfs, df])
    dfs = dfs.iloc[:,1:]
    du.insert_pd_tosql('pv_magazine', dfs)    



if __name__ == '__main__':
    pv_main()