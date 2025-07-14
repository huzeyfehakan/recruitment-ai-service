import json
import requests

def classify_skills(text):
    prompt = f"""
Aşağıdaki özgeçmiş metninde geçen becerileri 3 sınıfa ayır:
- tech: teknik beceriler (örneğin: Python, Java, React)
- soft: iletişim, liderlik gibi beceriler
- none: beceri olmayan ifadeler
// 2 sınıf, extract et ,  

Metin:
{text}

Çıktı formatı: JSON
[
  {{ "skill": "Python", "type": "tech" }},
  {{ "skill": "Takım çalışması", "type": "soft" }},
  ...
]
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    raw_response = response.json()["response"]
    try:
        start = raw_response.index("[")
        end = raw_response.rindex("]") + 1
        json_part = raw_response[start:end]
        return json.loads(json_part)
    except Exception as e:
        return {"error": "Failed to extract JSON", "raw": raw_response}