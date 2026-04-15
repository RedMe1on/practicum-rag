from curl_cffi import requests
from bs4 import BeautifulSoup
import time
import os

BASE_URL = "https://cyberpunk.fandom.com"
OUTPUT_DIR = "cyberpunk_pages"
PAGES_TO_COLLECT = 30

# Known article URLs to scrape
KNOWN_ARTICLES = [
    "/wiki/Cyberpunk_2077",
    "/wiki/V",
    "/wiki/Johnny_Silverhand",
    "/wiki/Night_City",
    "/wiki/Arasaka",
    "/wiki/Militech",
    "/wiki/Netrunning",
    "/wiki/Cyberware",
    "/wiki/Alt_Cunningham",
    "/wiki/Adam_Smasher",
    "/wiki/Judy_Alvarez",
    "/wiki/Panam_Palmer",
    "/wiki/Goro_Takemura",
    "/wiki/River_Ward",
    "/wiki/Kerry_Eurodyne",
    "/wiki/Rogue_Amendiares",
    "/wiki/Viktor_Vector",
    "/wiki/Misty_Olszewski",
    "/wiki/Dexter_DeShawn",
    "/wiki/Evelyn_Parker",
    "/wiki/Yorinobu_Arasaka",
    "/wiki/Saburo_Arasaka",
    "/wiki/Hanamako_Arasaka",
    "/wiki/Placide",
    "/wiki/Brick",
    "/wiki/Royce",
    "/wiki/Meredith_Stout",
    "/wiki/Sandra_Dorsett",
    "/wiki/T-Bug",
    "/wiki/Jackie_Welles",
]

# curl_cffi impersonates real browser TLS fingerprint (bypasses Cloudflare)
session = requests.Session(impersonate="chrome120")


def extract_text_from_page(url):
    """Extract only the text content from a wiki page, bypassing Cloudflare."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()

    # Remove tables (they're usually navigation/infoboxes)
    for table in soup.find_all("table"):
        table.decompose()

    # Remove infobox
    for infobox in soup.find_all("aside"):
        infobox.decompose()

    # Try to get main content
    content = soup.find("div", id="mw-content-text")
    if not content:
        content = soup.find("div", class_="mw-content-ltr")
    if not content:
        content = soup.find("main")
    if not content:
        return ""

    # Remove table of contents
    toc = content.find("div", id="toc")
    if toc:
        toc.decompose()

    # Get text and clean it up
    text = content.get_text(separator="\n", strip=True)

    # Clean up multiple newlines and whitespace
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Collecting {PAGES_TO_COLLECT} pages from {BASE_URL}...")
    print("Using curl_cffi with Chrome impersonation to bypass Cloudflare.\n")

    collected = 0
    for i, page_path in enumerate(KNOWN_ARTICLES):
        if collected >= PAGES_TO_COLLECT:
            break

        url = BASE_URL + page_path
        try:
            print(f"[{collected + 1}/{PAGES_TO_COLLECT}] Fetching: {page_path}")
            text = extract_text_from_page(url)

            if text and len(text) > 100:  # Only save pages with substantial content
                # Create filename from URL
                page_name = page_path.split("/wiki/")[-1].replace("/", "_")
                filename = os.path.join(OUTPUT_DIR, f"{page_name}.txt")

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"  Saved: {filename} ({len(text)} chars)")
                collected += 1
            else:
                print(f"  Skipped (too short or empty)")

            # Be polite to the server
            time.sleep(2)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nDone! Collected {collected} pages in '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()

