import requests

url = "http://127.0.0.0:8000/upload?session_id=test_session"
files = {'file': ('dummy.pdf', b'dummy content', 'application/pdf')}
response = requests.post(url, files=files)
print(response.status_code)
print(response.text)
