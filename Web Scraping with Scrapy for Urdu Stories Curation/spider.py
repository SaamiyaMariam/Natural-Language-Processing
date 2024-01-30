# Saamiya Mariam 20i-0612
# NLP Assignment 02
import scrapy, re
# re library for regular expressions

class I200612UrduStoriesSpiderSpider(scrapy.Spider):
    name = "i200612_urdu_stories_spider"
    allowed_domains = ["urduzone.net"]
    start_urls = ["https://urduzone.net"]

    def parse(self, response):

        # removing HTML tags and extracting text data
        text_data = response.css(' ').extract_first()

        # since we want to keep urdu text only, we will filter out non-urdu words
        urdu_text = re.findall(r'[\u0600-\u06FF\s]+', text_data)
        processed_urdu = " ".join(urdu_text)

        # used to output data from the spider. 
        # stores it in an internal buffer and yields it to the Scrapy framework to be processed and saved
        yield 
        {
            'urdu_text': processed_urdu
        }
        pass
