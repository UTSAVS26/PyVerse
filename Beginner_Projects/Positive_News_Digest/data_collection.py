import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_news():
    base_URL = 'https://www.goodnewsnetwork.org'
    for i in range(1,3):
        URL = f"{base_URL}/category/news/page/{i}/"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.find_all('div', class_="td-module-thumb")

        news_data = []
        for idx,article in enumerate(articles):

            title = article.find('a')['title']
            link = article.find('a')['href']

            article_url = link

            article_page = requests.get(article_url)
            article_soup = BeautifulSoup(article_page.content, 'html.parser')

            content_div = article_soup.find('div', class_="td-post-content")

            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([para.get_text(strip=True) for para in paragraphs])
                # Find the img tag within the figure
                img = article.find('a').find('img')
                if img and 'src' in img.attrs:
                    # Get the src attribute of the img tag
                    image_src = img['src']
                    short_src = image_src[:-12] + image_src[-4:]   #removing the thumbnail photo size part
            else:
                content = "Content not found"
                short_src = "Image not found"
            
            news_data.append({"id": idx, "Title":title, "Content":content, "Image": short_src})

    news_df = pd.DataFrame(news_data)

    news_df.to_csv('positive_news_dupl.csv',index=False)

    return news_df

# scrape_news()