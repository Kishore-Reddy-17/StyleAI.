import urllib.request, re

def get_bing_image(query):
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
        # Bing stores image URLs in murl&quot;:&quot;http...&quot;
        urls = re.findall(r'murl&quot;:&quot;(https?://.*?\.(?:jpg|jpeg|png))&quot;', html, re.IGNORECASE)
        return urls[:3] if urls else []
    except Exception as e:
        return str(e)

print(get_bing_image("mens navy blue formal shirt product photography isolated white background"))
