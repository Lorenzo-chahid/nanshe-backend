import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


class ImageAnalyzer:
    def __init__(self):
        self.model = "gpt-4"  # Utiliser GPT-4 pour les analyses

    def analyze_image(self, image_url: str) -> str:
        prompt = f"Analyze the following image and provide a detailed description of the physical appearance of the person in the image. URL: {image_url}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        analysis_result = response.choices[0].message["content"].strip()
        return analysis_result
