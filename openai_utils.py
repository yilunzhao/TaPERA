from openai import OpenAI, AzureOpenAI
import os

# Initialize client based on available environment variables
if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
    # Use Azure OpenAI
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    print("Using Azure OpenAI")
elif os.getenv("OPENAI_API_KEY"):
    # Use regular OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Using OpenAI")
else:
    raise ValueError("Please set either Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY) or OpenAI credentials (OPENAI_API_KEY)")

def get_completion(messages, model="gpt-35-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print("get_completion error:", e)
        return None

def get_function_completion(messages, functions=None, function_call=None, model="gpt-35-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=functions,
            tool_choice="auto",
        )
        return response.choices[0].message.tool_calls[0]
    except Exception as e:
        print("get_function_completion error:", e)
        return None
