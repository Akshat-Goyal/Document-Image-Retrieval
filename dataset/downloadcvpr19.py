import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

urls = [
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-16",
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-17",
    "https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-18",
]

# If there is no such folder, the script will create one automatically
folder_location = "./papers"
if not os.path.exists(folder_location):
    os.mkdir(folder_location)

responses = [requests.get(url) for url in urls]

soups = [BeautifulSoup(response.text, "html.parser") for response in responses]
links = []
for soup in soups:
    for link in soup.select("a[href$='.pdf']"):
        links.append(link)

for link in soup.select("a[href$='.pdf']"):
    filename = os.path.join(folder_location, link["href"].split("/")[-1])
    print("Downloading: ", link)
    with open(filename, "wb") as f:
        f.write(requests.get(urljoin(url, link["href"])).content)
