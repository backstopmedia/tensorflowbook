import requests
import os
from bs4 import BeautifulSoup

from helpers import ensure_directory


class ArxivAbstracts:

    ENDPOINT = 'http://export.arxiv.org/api/query'
    PAGE_SIZE = 100

    def __init__(self, cache_dir, categories, keywords, amount=None):
        self.categories = categories
        self.keywords = keywords
        cache_dir = os.path.expanduser(cache_dir)
        ensure_directory(cache_dir)
        filename = os.path.join(cache_dir, 'abstracts.txt')
        if not os.path.isfile(filename):
            with open(filename, 'w') as file_:
                for abstract in self._fetch_all(amount):
                    file_.write(abstract + '\n')
        with open(filename) as file_:
            self.data = file_.readlines()

    def _fetch_all(self, amount):
        page_size = type(self).PAGE_SIZE
        count = self._fetch_count()
        if amount:
            count = min(count, amount)
        for offset in range(0, count, page_size):
            print('Fetch papers {}/{}'.format(offset + page_size, count))
            yield from self._fetch_page(page_size, count)

    def _fetch_page(self, amount, offset):
        url = self._build_url(amount, offset)
        response = requests.get(url)
        soup = BeautifulSoup(response.text)
        for entry in soup.findAll('entry'):
            text = entry.find('summary').text
            text = text.strip().replace('\n', ' ')
            yield text

    def _fetch_count(self):
        url = self._build_url(0, 0)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        count = int(soup.find('opensearch:totalresults').string)
        print(count, 'papers found')
        return count

    def _build_url(self, amount, offset):
        categories = ' OR '.join('cat:' + x for x in self.categories)
        keywords = ' OR '.join('all:' + x for x in self.keywords)
        url = type(self).ENDPOINT
        url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
        url += '&max_results={}&offset={}'.format(amount, offset)
        return url