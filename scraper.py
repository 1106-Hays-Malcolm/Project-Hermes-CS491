import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
from os import listdir
from time import sleep
import json

# URLs and directories
ROOT_URL = "https://bg3.wiki/wiki"
DATA_DIRECTORY = "./data"
HTML_DIRECTORY = DATA_DIRECTORY + "/rawHTML"
JSON_DIRECTORY = DATA_DIRECTORY + "/json"
JSON_FILENAME = "/data.json"

# CSS class names for BeautifulSoup
QUEST_CLASS_NAME = "bg3wiki-imagetext"
QUEST_TEXT_CLASS_NAME = "bg3wiki-imagetext-text"
QUEST_OBJECTIVE_CLASSES = [
    "toccolours",
    "mw-collapsible",
    "mw-collapsed",
    "mw-made-collapsible"
]
QUEST_OBJECTIVE_TITLE_STYLE = "font-weight: bold; line-height: 1.6;"
QUEST_OBJECTIVE_LIST_CLASS = "mw-collapsible-content"
QUEST_WALKTHROUGH_SECTION_CLASS = "mw-headline"
QUEST_HEADINGS = [
    "Objectives",
    "Walkthrough",
    "Quest_Rewards",
    "Notes",
    "Achievements"
]

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


def parse_quest_objectives(soup):
    parsed_objectives = {}

    objectives = soup.find_all(True, {"class": QUEST_OBJECTIVE_CLASSES})
    for objective in objectives:
        title_html = objective.find("div", attrs={"style": QUEST_OBJECTIVE_TITLE_STYLE})
        objective_list_html = objective.find("div", class_=QUEST_OBJECTIVE_LIST_CLASS)

        if objective == -1 or title_html is None or objective_list_html is None:
            continue

        title = title_html.text.strip()
        objective_list = objective_list_html.find("ul")
        objective_list_cleaned = []

        if objective_list is None:
            one_objective = objective.find("p")
            if one_objective is not None:
                objective_list_cleaned.append(one_objective.text.strip())
        else:
            for item in objective_list:
                item = item.text.strip()
                if item != '':
                    objective_list_cleaned.append(item)

        parsed_objectives[title] = objective_list_cleaned

    return parsed_objectives


def parse_walkthrough(soup):
    parsed_walkthrough = []

    def is_walkthrough_step_heading(tag):
        try:
            return (tag.parent.name == "h3" or tag.parent.name == "h2") and \
                    tag.has_attr("class") and \
                    tag.name == "span" and \
                    QUEST_WALKTHROUGH_SECTION_CLASS in tag.get("class") and \
                    tag.get("id") not in QUEST_HEADINGS
        except AttributeError:
            return False

    def is_major_section_heading(tag):
        try:
            return tag.get("id") in QUEST_HEADINGS
        except AttributeError:
            return False

    def is_walkthrough_heading(tag):
        try:
            return tag.get("id") == "Walkthrough"
        except AttributeError:
            return False

    walkthrough_headings = soup.find_all(is_walkthrough_step_heading)

    # Some walkthroughs have no headings in the instructions
    if (walkthrough_headings == []):
        inner_text = ""
        next_sibling = soup.find(id="Walkthrough").parent.next_sibling
        while next_sibling is not None and not is_major_section_heading(next_sibling):
            inner_text += next_sibling.get_text()
            next_sibling = next_sibling.next_sibling
        parsed_walkthrough.append(["", inner_text])
        print(inner_text)

    # print()
    # print(walkthrough_headings)
    # print()


def main():
    # Create directories to store scraped data if they do not exist
    for p in [DATA_DIRECTORY, JSON_DIRECTORY, HTML_DIRECTORY]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # The page that lists all of the quests with links
    page = get_html("/Quests")

    soup = BeautifulSoup(page, "html.parser")

    filenames = get_all_downloaded_filenames(HTML_DIRECTORY)
    # for name in filenames:
    #     print(name)

    # Counter for downloads
    num_downloaded = 0
    num_skipped = 0

    quests = soup.find_all(class_=QUEST_CLASS_NAME)
    parsed_quests = {}
    for quest in quests:
        quest_text = quest.find(class_=QUEST_TEXT_CLASS_NAME)
        title_link = quest_text.find("a")
        if title_link == -1:
            continue

        # Get information from the individual links
        title_link_text = title_link.text.strip()
        title_link_url = title_link["href"]
        title_link_url_fixed = "/" + re.sub(r"/.*/", "", title_link_url)

        # Count the files downloaded and skipped
        if not SKIP_DOWNLOADED_FILES or title_link_url_fixed not in filenames:
            num_downloaded += 1
        else:
            num_skipped += 1

        quest_html = get_html(title_link_url_fixed, filenames)
        parsed_quests[title_link_text] = {}
        quest_soup = BeautifulSoup(quest_html, "html.parser")
        parsed_quests[title_link_text]["Objectives"] = parse_quest_objectives(quest_soup)
        print(title_link_text)
        parse_walkthrough(quest_soup)

    print("\n" + "=" * 50 + "\n")
    print("Total number of downloaded files: " + str(num_downloaded))
    print("Total number of files skipped for download: " + str(num_skipped))

    # with open(JSON_DIRECTORY + JSON_FILENAME, "w") as json_file:
    #     json_file.write(json.dumps(parsed_quests, indent=4))

if __name__ == "__main__":
    main()
