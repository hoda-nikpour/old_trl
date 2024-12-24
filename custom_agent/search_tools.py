
import re
import os
import json
import requests
from datetime import datetime

def log(desc, value):
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(desc, value)

def google_search_arxiv_id(query, num=10, end_date=None):
    url = "https://google.serper.dev/search"

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
        "q": search_query, 
        "num": num, 
        "page": 1, 
    })

    headers = {
        'X-API-KEY': 'your google keys',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        
        if response.status_code == 200:
            results = json.loads(response.text)
            
            # parse arxiv id
            arxiv_id_list, details = [], {}
            for paper in results['organic']:
                if "snippet" in paper:
                    cited_by = re.search(r'Cited by (\d+)', paper["snippet"]).group(0) if re.search(r'Cited by (\d+)', paper["snippet"]) else None
                arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1) if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]) else None
                if arxiv_id:
                    arxiv_id_list.append(arxiv_id)
                    details[arxiv_id] = {"arxiv_id": arxiv_id, "google_search_position": paper["position"], "cited_by": cited_by}
            return list(set(arxiv_id_list))
        
        else:
            print(f"Failed to request google. Status code: {response.status_code}")
            return None
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None
