import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
from os import listdir
from time import sleep

# URLs and directories
ROOT_URL = "https://bg3.wiki/wiki"
DATA_DIRECTORY = "./data"
HTML_DIRECTORY = DATA_DIRECTORY + "/rawHTML"
QUESTS_DIRECTORY = DATA_DIRECTORY + "/json"

# CSS class names for BeautifulSoup
QUEST_CLASS_NAME = "bg3wiki-imagetext"
QUEST_TEXT_CLASS_NAME = "bg3wiki-imagetext-text"

# To avoid downloading the wiki a lot of times
USE_DOWNLOADED_HTML = True
# To avoid downloading HTML pages that were already downloaded
SKIP_DOWNLOADED_FILES = True

# A short delay so you won't get locked out for making too many requests
REQUEST_DELAY_SECONDS = 5


# Get the names of all the files that have already been previously downloaded
def get_all_downloaded_filenames(path):
    filenames = listdir(path)
    filenames_no_ext = []

    # Put the slash in front of every file name
    for filename in filenames:
        filenames_no_ext.append("/" + str(Path(filename).with_suffix("")))

    return filenames_no_ext


def get_html(path, excluded_paths=[]):
    # If we are using the HTML that was previously downloaded
    if USE_DOWNLOADED_HTML:
        with open(HTML_DIRECTORY + path + ".html", "rb") as file:
            raw_html = file.read()

    # If we are downloading the HTML
    else:
        # If the file should be skipped for download and use the file on disk instead
        if not SKIP_DOWNLOADED_FILES or path not in excluded_paths:
            print("Now downloading: " + path)
            raw_html = requests.get(ROOT_URL + path).content
            sleep(REQUEST_DELAY_SECONDS)
            with open(HTML_DIRECTORY + path + ".html", "wb+") as file:
                file.write(raw_html)

        # If a new HTML file should be downloaded
        else:
            print("Skipping and using previously downloaded file: " + path)
            with open(HTML_DIRECTORY + path + ".html", "rb") as file:
                raw_html = file.read()

    return raw_html


def main():
    # Create directories to store scraped data if they do not exist
    for p in [DATA_DIRECTORY, QUESTS_DIRECTORY, HTML_DIRECTORY]:
        Path(p).mkdir(parents=True, exist_ok=True)

    page = get_html("/Quests")

    soup = BeautifulSoup(page, "html.parser")

    filenames = get_all_downloaded_filenames(HTML_DIRECTORY)
    # for name in filenames:
    #     print(name)

    num_downloaded = 0
    num_skipped = 0

    quests = soup.find_all(class_=QUEST_CLASS_NAME)
    for quest in quests:
        quest_text = quest.find(class_=QUEST_TEXT_CLASS_NAME)
        title_link = quest_text.find("a")
        if title_link == -1:
            continue

        # Get information from the individual links
        title_link_text = title_link.text.strip()
        title_link_url = title_link["href"]
        title_link_url_fixed = "/" + re.sub(r"/.*/", "", title_link_url)

        if not SKIP_DOWNLOADED_FILES or title_link_url_fixed not in filenames:
            num_downloaded += 1
        else:
            num_skipped += 1

        get_html(title_link_url_fixed, filenames)

    print("\n" + "=" * 50 + "\n")
    print("Total number of downloaded files: " + str(num_downloaded))
    print("Total number of files skipped for download: " + str(num_skipped))


if __name__ == "__main__":
    main()
