from question_answer_with_pdf import ConverseAPI, PDFProcessor
import gradio as gr

class GradioInterface:
    def __init__(self):
        self.api = ConverseAPI()
        self.prompts = {'Prompt:1':
            """Human: I'm going to give you a document. Then I'm going to ask you a question about it. I'd like you to first write down exact quotes of parts of the document that would help answer the question, and then I'd like you to answer the question using facts from the quoted content. Here is the document:
 
<document>{text}</document>
 
First, find the quotes from the document that are most relevant to answering the question, and then print them in numbered order. Quotes should be relatively short.
 
If there are no relevant quotes, write "No relevant quotes" instead.
 
Then, answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.
 
Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
 
<example>
<quotes>:
    <quote> [1] "Company X reported revenue of $12 million in 2021."</quote>
    <quote> [2] "Almost 90% of revene came from widget sales, with gadget sales making up the remaining 10%."</quote>
 </quotes>

<answer>
Company X earned $12 million. [1]  Almost 90% of it was from widget sales. [2]
</answer>
</example>
 
Here is the first question: {question}
 
If the question cannot be answered by the document, say so.
 
Answer the question immediately without preamble.
 
Assistant:""",
           'Prompt:2': """
                    Here is a document <doc> {text} </doc>. 
                    Answer this question using the provided doc only. 
                    If don't know say I don't have it.

                    {question}

                    answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.
                    Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
 
<example>
<quotes>:
   
 </quotes>

<answer>
Company X earned $12 million. [1]  Almost 90% of it was from widget sales. [2]
</answer>
</example>
                    """,
            'Prompt:3': """
                    Answer this question using the provided context only. If don't know say I don't have it.

                    Context: {text}

                    {question}
                    """
        }
        self.model_ids = [
            "anthropic.claude-v2:1",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ]

    def process_file(self, file, question, prompt_key , model_id):

        prompt_message = self.prompts[prompt_key]
        text = PDFProcessor.get_text_from_pdf(file)
        return self.api.ask_question(text, question, prompt_message, model_id)

    def launch(self):
        prompts = [f"{key}" for key in self.prompts.keys()]
        with gr.Blocks() as demo:
            with gr.Row():
                pdf_file = gr.File(label="Upload PDF")
                question = gr.Textbox(label="Enter your question")
            with gr.Row():
                prompt_dropdown = gr.Dropdown(prompts, label="Select a prompt")
                model_id_dropdown = gr.Dropdown(self.model_ids, label="Select a model ID")
            with gr.Row():
                submit_btn = gr.Button("Submit")
            with gr.Row():
                selected_prompt = gr.Textbox(label="Selected Prompt", lines=10)
                raw_output = gr.Textbox(label="Raw Output", lines=10)
            with gr.Row():
                quotes_output = gr.Textbox(label="Quotes", lines=10)
                answer_output = gr.Textbox(label="Answer", lines=10)
            
            submit_btn.click(fn=self.process_file, inputs=[pdf_file, question, prompt_dropdown, model_id_dropdown], outputs=[selected_prompt, raw_output, quotes_output, answer_output])
        
        demo.launch()

# Main execution
if __name__ == "__main__":
    gradio_interface = GradioInterface()
    gradio_interface.launch()