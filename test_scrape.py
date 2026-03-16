import urllib.request, re

def get_amazon_img(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
        matches = re.findall(r'\"(https://m\.media-amazon\.com/images/I/[A-Za-z0-9\-_]+\.jpg)\"', html)
        return matches[0] if matches else None
    except Exception as e:
        return str(e)

print(get_amazon_img('https://www.amazon.in/s?k=navy+blue+shirt'))
