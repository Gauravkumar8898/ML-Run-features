import openai
import mlrun

# Define the MLRun function
@mlrun.handler()
def openais(prompt: str, context):
    # Retrieve the OpenAI API key from MLRun secrets
    api_key = mlrun.secrets.get_secret_or_env("OPEN_API_KEY")
    openai.api_key = api_key

    # Make the API call
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=60,
        api_key="OPEN_API_KEY"

    )
    context.log_result(key="response", value=response.choices[0].text.strip())
    # return response.choices[0].text.strip()




