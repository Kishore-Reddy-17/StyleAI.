import urllib.request, urllib.parse, re

def get_bing_image(query):
    search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
    req_bing = urllib.request.Request(search_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    try:
        html = urllib.request.urlopen(req_bing, timeout=5).read().decode('utf-8', errors='ignore')
        img_matches = re.findall(r'murl&quot;:&quot;(https?://.*?\.(?:jpg|jpeg|png|webp))&quot;', html, re.IGNORECASE)
        print("Found:", len(img_matches), img_matches[:3])
    except Exception as e:
        print(f"Error: {e}")

get_bing_image("mens Navy Blue Solid Formal Shirt product photography isolated white")
