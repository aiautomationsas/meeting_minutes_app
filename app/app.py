import asyncio

from langgraph_sdk import get_client

url = "https://ht-rash-variability-38-164af1dbe40454e9bd46e5f19b24bc7c.default.us.langgraph.app"
client = get_client(url=url)
assistant_id = "agent"

async def main():
    # Create thread
    thread = await client.threads.create()
    print(f"Created thread: {thread}")

    input_data = {
        "audioFile": "",
        "transcript": "Crea un acta ficticia de na reunión de desarrollo de software. No tengo la transcripción",
        "wordCount": 500,
        "critique": "",
        "approved": False,
        "minutes": "",
        "outputFormatMeeting": "",
        "messages": []
    }

    async for chunk in client.runs.stream(
        thread['thread_id'],
        assistant_id,
        input=input_data,
        interrupt_before=["human_critique"],
        stream_mode="values"
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
