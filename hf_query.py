import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "user",
            "content": "give me a pre trained ml model that can detect skin tone gender and could discriminate male and female separately for my project which is based on fashion recommendation system"
        }
    ],
)

print(completion.choices[0].message)
