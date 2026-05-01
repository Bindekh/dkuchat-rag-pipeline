# import sys
# import os
# import argparse
# sys.path.append(".")

# import gradio as gr
# from agent.agent import RAGAgent, setup_dspy
# from dotenv import load_dotenv
# load_dotenv()

# # --- CLI args ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default=None, help="LLM model name")
# parser.add_argument("--base-url", default=None, help="LLM server base URL")
# parser.add_argument("--port", type=int, default=7860)
# args, _ = parser.parse_known_args()

# # --- Init ---
# active_model = setup_dspy(model_name=args.model, base_url=args.base_url)
# agent = RAGAgent()

# # --- Helpers ---

# def format_sources(sources: list) -> str:
#     if not sources:
#         return "No sources found."
#     lines = []
#     seen = set()
#     for s in sources:
#         key = (s["source"], s["page"])
#         if key not in seen:
#             seen.add(key)
#             if s["method"] == "internet":
#                 lines.append(f"• [{s['source']}]({s['source']})")
#             else:
#                 lines.append(f"• **{s['source']}** — page {s['page']}")
#     return "\n".join(lines)


# def query(question: str):
#     if not question.strip():
#         return "Please enter a question. / 请输入问题。", "", ""
#     try:
#         result = agent(question=question)
#         sources_text = format_sources(result.sources)
#         tools_text = ", ".join(result.tools_used)
#         return result.answer, sources_text, tools_text
#     except Exception as e:
#         return f"❌ Error: {e}", "", ""


# # --- UI ---

# with gr.Blocks(title="DKU Mini Chatbot", theme=gr.themes.Soft(
#         primary_hue="blue",
#         secondary_hue="indigo",
#         neutral_hue="slate"
#     )) as demo:
#     gr.Markdown(
#          f"# Mini DKU Chatbot\n"
#         f"**Model:** `{active_model}`"
#     )

#     with gr.Row():
#         with gr.Column(scale=2):
#             question_box = gr.Textbox(
#                 label="Your question / 你的问题",
#                 placeholder="e.g. What are the graduation requirements? / 毕业要求是什么？",
#                 lines=3
#             )
#             submit_btn = gr.Button("Ask / 提问", variant="primary")

#     with gr.Row():
#         with gr.Column(scale=2):
#             answer_box = gr.Markdown(label="Answer / 回答")
#         with gr.Column(scale=1):
#             sources_box = gr.Markdown(label="Sources / 来源")
#             tools_box = gr.Textbox(
#                 label="Tools used / 使用工具",
#                 interactive=False
#             )

#     submit_btn.click(
#         fn=query,
#         inputs=question_box,
#         outputs=[answer_box, sources_box, tools_box]
#     )
#     question_box.submit(
#         fn=query,
#         inputs=question_box,
#         outputs=[answer_box, sources_box, tools_box]
#     )

# if __name__ == "__main__":
#     demo.launch(share=False, server_port=args.port)


import sys
import os
import argparse
sys.path.append(".")

import gradio as gr
from agent.agent import RAGAgent, setup_dspy
from dotenv import load_dotenv
load_dotenv()

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="LLM model name")
parser.add_argument("--base-url", default=None, help="LLM server base URL")
parser.add_argument("--port", type=int, default=7860)
args, _ = parser.parse_known_args()

# --- Init ---
active_model = setup_dspy(model_name=args.model, base_url=args.base_url)
agent = RAGAgent()

# --- Helpers ---

def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    seen = set()
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            if s["method"] == "internet":
                lines.append(f"• [{s['source']}]({s['source']})")
            else:
                lines.append(f"• **{s['source']}** — page {s['page']}")
    return "\n".join(lines)


def query(question: str, history: list):
    if not question.strip():
        return history, ""
    try:
        result = agent(question=question)
        sources_text = format_sources(result.sources)
        tools_text = ", ".join(result.tools_used)

        full_answer = result.answer
        if sources_text:
            full_answer += f"\n\n**Sources:**\n{sources_text}"
        if tools_text:
            full_answer += f"\n\n*Tools: {tools_text}*"

    except Exception as e:
        full_answer = f"❌ Error: {e}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer}
    ]
    return history, ""


def clear_chat():
    return [], ""


# --- UI ---

with gr.Blocks(title="DKU Mini Chatbot") as demo:
    gr.Markdown(
        f"# Mini DKU Chatbot\n"
        f"**Model:** `{active_model}`"
    )

    chatbot = gr.Chatbot(
        label="Conversation / 对话",
        height=450
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question / 你的问题",
            placeholder="e.g. What are the graduation requirements? / 毕业要求是什么？",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("Ask / 提问", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Chat / 清除对话", variant="secondary")

    submit_btn.click(
        fn=query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box]
    )

    question_box.submit(
        fn=query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, question_box]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=args.port,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate"
        )
    )
