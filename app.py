import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question
from implementation.llm import prepare_local_model

load_dotenv(override=True)


def message_text(message) -> str:
    """Gradio 6 stores message content as a list of text parts."""
    content = message["content"] if isinstance(message, dict) else message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content)


def format_context(context):
    if not context:
        return (
            "No context was retrieved. The knowledge base has no documents yet. "
            "Run ingestion before expecting answers from the Insurellm files."
        )
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    question = message_text(history[-1])
    prior = [
        {
            "role": item["role"] if isinstance(item, dict) else item.role,
            "content": message_text(item),
        }
        for item in history[:-1]
    ]
    try:
        answer, context = answer_question(question, prior)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"The request failed: {exc}"})
        return history, f"The request failed: {exc}"
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    prepare_local_model()

    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                    buttons=["copy"],
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme, show_error=True)


if __name__ == "__main__":
    main()
