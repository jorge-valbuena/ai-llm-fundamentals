"""
Exercise: Running a Local LLM with LM Studio
=============================================
Instead of sending prompts to a cloud API, you can run an LLM entirely
on your own machine using LM Studio (https://lmstudio.ai).

LM Studio exposes an OpenAI-compatible server, so we can use the same
`openai` Python library — just pointed at localhost instead of Azure.

Setup:
  1. Download and open LM Studio.
  2. Download a model (e.g. liquid/lfm2.5-1.2b).
  3. Go to the "Developer" tab and start the local server.
  4. The server runs at http://127.0.0.1:1234 by default.

Key difference vs cloud:
  - No API key needed (set to any non-empty string).
  - No internet connection required after the model is downloaded.
  - Speed depends on your local hardware (CPU/GPU).
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# LM Studio server — change these in .env if needed
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_LLM_MODEL    = os.getenv("LOCAL_LLM_MODEL", "lfm2.5-1.2b-instruct")

# LM Studio doesn't require a real API key, but the client expects one
client = OpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key="lm-studio"
)


def ask_local_llm(user_input: str) -> None:
    print(f"\n--- Local LLM Demo (model: {LOCAL_LLM_MODEL}) ---")
    print(f"Input: \"{user_input}\"")

    system_prompt = """
        You are an intelligent, proactive assistant dedicated to helping you organize, prioritize, and execute daily activities with precision and efficiency. 
        Your role is to streamline your time, enhance productivity, and support your goals across multiple domains such as work, personal development, health, finances, and more.

        #### **Core Functions:**
        1. **Task Management:**
        - Create, update, and track daily, weekly, and monthly tasks.
        - Set priorities, deadlines, and reminders for all tasks.
        - Suggest optimal time blocks using your schedule and energy patterns.

        2. **Planning & Organization:**
        - Generate actionable to-do lists based on goals and deadlines.
        - Create calendars (digital or physical) aligned with your priorities.
        - Integrate reminders for appointments, meetings, and important events.

        3. **Information Retrieval & Updates:**
        - Provide concise summaries of current events, news, and market trends.
        - Offer recommendations based on your preferences and past behavior.
        - Keep you informed without overwhelming you with unnecessary details.

        4. **Personal Development:**
        - Suggest learning resources (articles, videos, podcasts) for skill development.
        - Track progress toward personal or professional goals.
        - Provide motivation and encouragement throughout the day.

        5. **Health & Wellness Support:**
        - Offer daily wellness tips, exercise routines, and nutrition advice.
        - Monitor sleep patterns and suggest improvements.
        - Help manage stress with mindfulness exercises or breathing techniques.

        6. **Financial Management (if applicable):**
        - Track expenses and budgeting goals.
        - Provide financial insights or alerts for savings opportunities.
        - Suggest investment strategies or cost-saving tips.

        7. **Customization & Adaptability:**
        - Learn your preferences over time to personalize task suggestions.
        - Adjust priorities dynamically based on changing circumstances.
        - Maintain a balanced approach between work, rest, and play.

        #### **Constraints & Best Practices:**
        - Always provide clear, concise, and actionable responses.
        - Prioritize accuracy and relevance in all information provided.
        - Avoid redundant or repetitive tasks unless you explicitly request them.
        - Respect your time and focus—optimize for efficiency without burnout.
    """

    response = client.chat.completions.create(
        model=LOCAL_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_input}
        ]
    )

    print(f"Response:\n{response.choices[0].message.content}")
    print(f"\n[Tokens used — prompt: {response.usage.prompt_tokens}, "
          f"completion: {response.usage.completion_tokens}]")


def list_local_models() -> None:
    """Print all models currently available in the LM Studio server."""
    print("\n--- Models available in LM Studio ---")
    models = client.models.list()
    for m in models.data:
        print(f"  - {m.id}")


def chat_loop() -> None:
    print("\nType your question (type 'bye' to exit).")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() == "bye":
            print("Goodbye!")
            break
        ask_local_llm(user_input)


if __name__ == "__main__":
    list_local_models()
    chat_loop()