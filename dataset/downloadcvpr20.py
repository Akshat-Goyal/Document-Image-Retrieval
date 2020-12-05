import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from rich.progress import track

urls = [
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-16",
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-17",
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-18",
]

folder_location = "./papers"

if not os.path.exists(folder_location):
    os.mkdir(folder_location)

responses = [requests.get(url) for url in urls]
dictonary = {}

soups = [BeautifulSoup(response.text, "html.parser") for response in responses]
links = []
for i,soup in enumerate(soups):
    for link in soup.select("a[href$='.pdf']"):
        links.append(link)
        dictonary[link] = urls[i]

for link in track(links):
    print(link)
    filename = os.path.join(folder_location, link["href"].split("/")[-1])
    with open(filename, "wb") as f:
        f.write(requests.get(urljoin(dictonary[link], link["href"])).content)
