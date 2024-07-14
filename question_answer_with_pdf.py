
import boto3
from botocore.exceptions import ClientError
from pypdf import PdfReader

class ConverseAPI:
    def __init__(self, region_name="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def ask_question(self, text, question, prompt, model_id):
        user_message = prompt.format(text=text, question=question)

        conversation = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]

        try:
            response = self.client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 2048, "stopSequences": ["\n\nHuman:"], "temperature": 0.5, "topP": 1},
                additionalModelRequestFields={"top_k": 250}
            )

            response_text = response["output"]["message"]["content"][0]["text"]
            quotes, answer = self.parse_response(response_text)
            return prompt, response_text, quotes, answer

        except (ClientError, Exception) as e:
            return f"ERROR: Can't invoke '{model_id}'. Reason: {e}", ""

    def parse_response(self, response_text):
        quotes_start = response_text.find("<quotes>") + len("<quotes>")
        quotes_end = response_text.find("</quotes>")
        answer_start = response_text.find("<answer>") + len("<answer>")
        answer_end = response_text.find("</answer>")
        
        quotes = response_text[quotes_start:quotes_end].strip()
        answer = response_text[answer_start:answer_end].strip()
        return quotes, answer


class PDFProcessor:
    @staticmethod
    def get_text_from_pdf(file):
        reader = PdfReader(file)
        text = ' '.join(page.extract_text() for page in reader.pages)
        return text

