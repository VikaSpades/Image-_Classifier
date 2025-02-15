import openai

API_KEY = open("OPENAI_API","r").read()
openai.api_key = API_KEY

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"user","content":"Cat"}
    ],
    stream=True
)
print(response)





