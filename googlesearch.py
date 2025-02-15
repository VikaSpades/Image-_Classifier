import requests
API_KEY = open('GOOGLE_SEARCH_API_KEY').read()
Search_Engine_ID = open('Search_Engine_ID').read()
search_query = 'blah'

url = 'https://www.googleapis.com/customsearch/v1'
params = {
    'q': search_query,
    'key': API_KEY,
    'cx': Search_Engine_ID,
    'searchType': 'image'
}
response = requests.get(url, params=params)
results = response.json()['items']

for item in results:
    print(item['link'])