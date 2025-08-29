import os
from dotenv import load_dotenv
from openai import AzureOpenAI

#Kernel references
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

def main():

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # Initialize the kernel
    kernel = Kernel()

    try:
        # Get configuration settings
        load_dotenv()
        open_ai_endpoint = os.getenv("OPEN_AI_ENDPOINT")
        open_ai_key = os.getenv("OPEN_AI_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        search_url = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = os.getenv("INDEX_NAME")


        # Get an Azure OpenAI chat client
        chat_client = AzureOpenAI(
            api_version = "2024-12-01-preview",
            azure_endpoint = open_ai_endpoint,
            api_key = open_ai_key
        )

        kernel.add_service(chat_client)
        setup_logging()
        logging.getLogger("kernel").setLevel(logging.DEBUG)
        # Create a history of the conversation
        history = ChatHistory()
        
        # Initialize prompt with system message
        prompt = [
            {"role": "system", "content": "You are a knowledgeable assistant that provides contextually relevant information based on the user's query, leveraging a combination of predefined knowledge and external data retrieval."}
        ]

        # Loop until the user types 'quit'
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Add the user input message to the prompt
            prompt.append({"role": "user", "content": input_text})

            history.add_user_message(input_text)
            # Additional parameters to apply RAG pattern using the AI Search index
            rag_params = {
                "data_sources": [
                    {
                        # he following params are used to search the index
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_url,
                            "index_name": index_name,
                            "authentication": {
                                "type": "api_key",
                                "key": search_key,
                            },
                            # The following params are used to vectorize the query
                            "query_type": "vector",
                            "embedding_dependency": {
                                "type": "deployment_name",
                                "deployment_name": embedding_model,
                            },
                        }
                    }
                ],
            }

            # Submit the prompt with the data source options and display the response
            response = chat_client.chat.completions.create(
                model=chat_model,
                messages=prompt,
                extra_body=rag_params
            )
            completion = response.choices[0].message.content
            history.add_message(completion)
            print(completion)

            # Add the response to the chat history
            prompt.append({"role": "assistant", "content": completion})

    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    main()