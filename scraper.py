import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
from os import listdir

# URLs and directories
ROOT_URL = "https://bg3.wiki/wiki"
DATA_DIRECTORY = "./data"
HTML_DIRECTORY = DATA_DIRECTORY + "/rawHTML"
QUESTS_DIRECTORY = DATA_DIRECTORY + "/quests"

# CSS class names for BeautifulSoup
QUEST_CLASS_NAME = "bg3wiki-imagetext"
QUEST_TEXT_CLASS_NAME = "bg3wiki-imagetext-text"

# To avoid downloading the wiki a lot of times
USE_DOWNLOADED_HTML = True
# To avoid downloading HTML pages that were already downloaded
SKIP_DOWNLOADED_FILES = True


def get_all_downloaded_filenames(path):
    filenames = listdir(path)
    filenames_no_ext = []

    for filename in filenames:
        filenames_no_ext.append("/" + str(Path(filename).with_suffix("")))

    return filenames_no_ext


def get_html(path, excluded_paths=[]):
    if USE_DOWNLOADED_HTML:
        with open(HTML_DIRECTORY + path + ".html", "rb") as file:
            raw_html = file.read()
    else:
        print("Now downloading: " + path)
        raw_html = requests.get(ROOT_URL + path).content
        with open(HTML_DIRECTORY + path + ".html", "wb+") as file:
            file.write(raw_html)

    return raw_html


def main():
    # Create directories to store scraped data if they do not exist
    for p in [DATA_DIRECTORY, QUESTS_DIRECTORY, HTML_DIRECTORY]:
        Path(p).mkdir(parents=True, exist_ok=True)

    page = get_html("/Quests")

    soup = BeautifulSoup(page, "html.parser")

    filenames = get_all_downloaded_filenames(HTML_DIRECTORY)
    for name in filenames:
        print(name)

    quests = soup.find_all(class_=QUEST_CLASS_NAME)
    for quest in quests:
        quest_text = quest.find(class_=QUEST_TEXT_CLASS_NAME)
        title_link = quest_text.find("a")
        if title_link == -1:
            continue

        title_link_text = title_link.text.strip()
        title_link_url = title_link["href"]
        title_link_url_fixed = "/" + re.sub(r"/.*/", "", title_link_url)

        # get_html(title_link_url_fixed)


if __name__ == "__main__":
    main()
