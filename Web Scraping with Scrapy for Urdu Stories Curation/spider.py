import scrapy

class UrduStoriesSpider(scrapy.Spider):
    name = 'I200612urdu_stories_spider'
    start_urls = ['https://www.urduzone.net']

    def parse(self, response):
        # Extract story links from the main page
        story_links = response.xpath('//a[@rel="bookmark"]/@href').extract()
        for link in story_links:
            yield response.follow(link, self.parse_story)

    def parse_story(self, response):
        # Extract the story title and text
        title = response.xpath('//title/text()').get()
        story_text = response.xpath('//p[@dir="rtl"]/text()').getall()
        story_text = ' '.join(story_text)  # Joining if the text is in multiple parts

        # Optional: Clean the title to remove extra text
        title = title.split('|')[0].strip() if title else title

        yield {
            'title': title,
            'story': story_text
        }

