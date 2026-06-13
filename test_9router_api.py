from openai import OpenAI
import os

# Connect to your local 9Router server
client = OpenAI(
    api_key=os.environ.get("NINEROUTER_API_KEY", "dummy-local-key"),
    base_url="http://127.0.0.1:20128/v1"
)

# Pick one of your available models
MODEL = "ag/gpt-oss-120b-medium"

try:
    print("Sending test request...\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "explain LLM in 1 sentence."
            }
        ],
        temperature=0.7,
    )

    print("=== SUCCESS ===\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("\n=== ERROR ===")
    print(type(e).__name__)
    print(e)
