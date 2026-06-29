"""
Реализован парсинг определенных столбцов таблицы, который в результате представлен JSON-файлом
"""
import scrapy
from scrapy.crawler import CrawlerProcess


class SechenovDirectSpider(scrapy.Spider):
    name = "nabludenie"

    def __init__(self, id_maj):
        self.id_maj = id_maj
        self.base_url = (
            "https://priem.sechenov.ru/local/components/firstbit/competition.list/templates/.default/applications.php"
            f"?COMPETITIVE_GROUP_ID={self.id_maj}"
            "&ADMISSION_LISTS=N&CONTRACT_IS_PAID=N&ORIGINAL_DOCUMENT=N"
            "&lang=ru&search=&highest_passing_priority=&highest_primary_priority=&header_consent="
        )

    def start_requests(self):
        yield self.make_request(page=1)

    def make_request(self, page):
        url = f"{self.base_url}&appPage_{self.id_maj}=page-{page}"
        return scrapy.Request(
            url=url,
            callback=self.parse,
            cb_kwargs={'page': page}
        )

    def parse(self, response, page):
        rows = response.css('tr')
        found = False

        for row in rows:
            words = row.xpath('string(.)').get().split()
            for i in range(len(words)-1, -1, -1):
                if words[i].isdigit():
                    idx = i
                    break


            if len(words) > 2 and words[1].isdigit() and len(words[1]) == 7:
                found = True
                yield {
                    'number': words[0],
                    'uid': words[1],
                    'total_score': words[2],
                    'status': ' '.join(words[idx+1:])
                }

        if found:
            yield self.make_request(page + 1)


if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "FEEDS": {
            "results_7_1.json": {
                "format": "json",
                "encoding": "utf8",
                "overwrite": True,
                "item_export_kwargs": {"ensure_ascii": False}
            },
        },
    })

    process.crawl(SechenovDirectSpider, id_maj=19591)
    process.start()
