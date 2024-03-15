from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
sys.path.append("../utils/pytube")
from pytube import Channel,YouTube #키워드 -> url
import time 
from bs4 import BeautifulSoup


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

class instance_search_expanding():
    def __init__(self, query):
        self.embed_html = self.get_youtube_resource(query)
        self.img_list = self.get_image_resource(query)

    def get_youtube_resource(self, query):
        query = query + ' youtube'
        url= 'https://www.google.com/search?q={}&tbm=vid&source=lnms'.format(query.replace(' ','+'))
        driver = webdriver.Chrome( options=global_obj.chrome_options)
        driver.delete_all_cookies()
        driver.get(url) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        div_tags = soup.find_all('div', id = 'search')
        a_tag = div_tags[0].find_all('a')
        youtube_list = []
        for a in a_tag:
            if 'href' in a.attrs:
                if 'youtube' in a.attrs['href']:
                    if 'google' not in a.attrs['href']:
                        if 'watch' in a.attrs['href']:
                            youtube_list.append(a.attrs['href'])
        youtube_list =  list(set(youtube_list))
        youtube_list = [self.get_youtube_metadata(i) for i in youtube_list[:6]]
        unique_dict = {item['thumbnail_url']: item for item in youtube_list}.values()
        unique_list = list(unique_dict)
        
        embed_html = []
        for url in unique_list:
            text = f'''<a href="{url['url']}"> <img width="300" height="180" src="{url['thumbnail_url']}"> </a>'''
            embed_html.append(text)
        return embed_html[:3]    
        
    def get_youtube_metadata(self, url):
        yt = YouTube(url)
        video_info = {
            "url":url,
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown",
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
            "embed_url" : yt.embed_url
        }
        return video_info
        
    def get_image_resource(self, query):
        url= 'https://www.google.com/search?q={}&tbs=qdr:6m&tbm=isch&source=lnms'.format(query.replace(' ','+'))
        driver = webdriver.Chrome( options=global_obj.chrome_options)
        driver.delete_all_cookies()
        driver.get(url) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        div_tags = soup.find_all('div')
        img_list = []
        href_list = []
        
        for idx, div in enumerate(div_tags):
            img_tags = soup.find_all(['img', 'a'])
            for t in img_tags:
                if 'src' in t.attrs:
                    if 'alt' in t.attrs:
                        if t.attrs['alt'] != '':
                            img_list.append(t.attrs['src'])
                if 'href' in t.attrs:
                    if 'google' not in t.attrs['href'] :
                        if 'http' in t.attrs['href'] :
                            href_list.append(t.attrs['href'])                    
            break

        img_html = []
        for href, img in zip(href_list, img_list):
            text = f'''<a href="{href}"> <img src="{img}"> </a>'''
            img_html.append(text)
            
        return img_html[:3]


if __name__ == '__main__':
    ise = instance_search_expanding(query = 'IQ8 microinverter price')