from openai import OpenAI

client = OpenAI(
    api_key="YOUR_NVIDIA_API_KEY",
    base_url="https://integrate.api.nvidia.com/v1",
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "what is LLM? explain in 1 sentence"
        }
    ],
    max_tokens=100,
)

print(resp.choices[0].message.content)
print(resp.usage)