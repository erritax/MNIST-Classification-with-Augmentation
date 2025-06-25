from google import genai

client = genai.Client(api_key = "ENTER KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)