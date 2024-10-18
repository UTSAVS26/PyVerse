## Automation Tool for Web Scraping using Selenium

### 🎯 Goal

We will be making a list of disney movies from IMDB scraped using Selenium. [IMDB]("https://www.imdb.com/list/ls026785255/") is one of the modern websites that uses javascript for dynamic loading, making it hard for beautifulSoup to scrape, But Selenium got us covered. 
---
### 🧾 Description

*Selenium* is a powerful python library for automation of web browser
You can install it using
`pip install selenium`
or if you are using anaconda
`conda install -c conda-forge selenium`.

Then you need to download a web-driver, in chrome it's ChromeDriver, in firefox it's geckodriver.
Here you will get the links to drivers https://pypi.org/project/selenium/ .

In my case I am using chromedriver (Remember the version of your chrome driver should match up with the version of chrome you are using)
and once can use any web browser like firfox, safari, edge, of your choice...

As for chromedriver, you can go to `https://chromedriver.chromium.org/downloads`


---
### 🧮 Features Implemented

Scrapes Disney movie's  
- Movie Title
- Trailer Link / Video
- Cover Image
- List of all Genres of Movie
- Synopsis - Short Description
- StoryLine of Movie

Extras
- Added loader to scroll to bottom of page to load all contents (that loads thorugh js)
- Added dialog box dismis logic to close any unexpectedly opened dialog for ratings etc that hinders with element interaction

---
### 📚 Libraries Needed

- **Selenium**: To get browser driver to interact with element of the web page

---
### 📊 Example Output:

```structure
[
    {
        'name': NAME OF DISNEY MOVIE,
        'video': URL OF TRAILER,
        'cover_img': COVER IMAGE,
        'genres': [
            {
                'name': NAME OF GENRE, 
                'url': IMDB LINK OF GENRE
            },
        ],
        'synopsis': SHORT SUMMARY,
        'story': STORYLINE'
    }
]
...,...,...
```

**Nidhi**  
[![GitHub](https://img.shields.io/badge/github-%2312100E.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nidhi2026) | [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nidhi-845150271/)