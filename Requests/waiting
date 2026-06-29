"""
Код для отслеживания обновления файла на сайте
"""
import requests
import hashlib
import time
from bs4 import BeautifulSoup
from datetime import datetime
import winsound

url = 'https://www.sechenov.ru/admissions/priemnaya-kampaniya-2024/svedeniya-o-zachislenii/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}




def play_sound():
    frequency = 1000  # Частота звука (в герцах)
    duration = 1000  # Продолжительность звука (в миллисекундах)
    winsound.Beep(frequency, duration)


def get_page_content(url):
    response = requests.get(url, headers=headers)
    return response.content


def extract_file_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    file_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
    return file_links


def check_for_updates(url, previous_links):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    current_content = get_page_content(url)
    current_links = extract_file_links(current_content)
    if set(current_links) != set(previous_links):
        print("Обнаружено обновление на сайте!")
        play_sound()
        return current_links
    else:
        print("Обновлений не обнаружено.")

        return previous_links



previous_links = extract_file_links(get_page_content(url))


while True:
    previous_links = check_for_updates(url, previous_links)
    time.sleep(60)  # Проверка обновлений каждый час
