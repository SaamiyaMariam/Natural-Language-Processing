# Urdu Stories Scrapy Spider

## Introduction
This Scrapy spider is designed to crawl and scrape Urdu stories from the website [UrduZone](https://www.urduzone.net). The spider extracts story links from the main page and then proceeds to fetch individual stories, capturing their titles and text.

## Spider Details
- **Spider Name**: I200612urdu_stories_spider
- **Start URL**: [https://www.urduzone.net](https://www.urduzone.net)

## How it Works
1. The spider starts by visiting the main page and extracting story links using XPath.
2. For each story link, it follows the link and extracts the title and text of the story.
3. The story text is cleaned and joined if it appears in multiple parts.
4. Optionally, the title is cleaned to remove extra text.

## Usage
To run the spider, use the following command in the terminal:

```bash
scrapy crawl I200612urdu_stories_spider -o output.json
```
## Output
The spider yields a dictionary for each story with the following structure:

```bash
{
    "title": "Story Title",
    "story": "Story Text..."
}
```
## Dependencies
Make sure to have Scrapy installed. If not, install it using:

```bash
pip install scrapy
```
