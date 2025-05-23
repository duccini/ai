import urllib.request

def download_and_read_text(url):
    print("Data loader: retrive the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
      raw_text = f.read()
      print("Total number of characters:", len(raw_text))
      print("### First 99 characters ###")
      print(raw_text[:99])

    return raw_text
