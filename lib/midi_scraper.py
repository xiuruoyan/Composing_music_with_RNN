import os
import re
import requests
import urllib

from bs4 import BeautifulSoup


midi_domain = "http://www.piano-midi.de"

def get_midi_links_from_page(midi_page):
    # Each artist page has a table for each album
    # Fetch only the actual composition midi links
    soup = BeautifulSoup(requests.get(midi_page).content, "html5")
    tables = soup.findAll("table", attrs={"class": "midi"})
    links = []
    for table in tables:
        links += table.findAll("a", attrs={"href": re.compile("midis\/")})
    return [link['href'] for link in links if not re.compile(r"format[0-9]").search(link['href'])]

# Go to classical midi page and get all the nav links for each artist
soup = BeautifulSoup(requests.get(midi_domain).content, "html5")
nav_links_soup = soup.find("div", attrs={"class": "navileft"})
artist_links = nav_links_soup.findAll("a")[1:-2]

# For each artist link, get the full artist url and retrieve all the midi links
midi_links = []
for link in artist_links:
    artist_url = urllib.request.urljoin(midi_domain, link['href'])
    midi_links += get_midi_links_from_page(artist_url)

for link in midi_links:
    # Form the full url
    dl_url = urllib.request.urljoin(midi_domain, link)

    # Store the file locally in the same path as the website
    # If the composer directory doesn't exist yet, create it
    file_loc = link
    dir_loc = os.path.dirname(file_loc)
    dir_loc = os.path.join('../data',dir_loc)
    file_loc = os.path.join('../data',file_loc)
    if not os.path.exists(dir_loc):
        os.makedirs(dir_loc)
    urllib.request.urlretrieve(dl_url, file_loc)
