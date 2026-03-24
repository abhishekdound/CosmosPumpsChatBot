import os
import logging
import base64
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import cloudinary
import cloudinary.uploader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

class ImageAnalyzer:
    """
    A professional-grade image extraction and analysis class for RAG pipelines.
    Combines heuristic filtering with VLM (Vision LLM) context-aware captioning.
    """

    def __init__(self):
        load_dotenv(override=True)
        print("GEMINI KEY LOADED:", os.getenv('GEMINI_API_KEY')[:15])
        self.vlm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        self.noise_patterns = [
            "logo", "icon", "sprite", "banner", "ads", "spacer", "pixel",
            "loader", "placeholder", "social", "button", "nav", "menu"
        ]
        self.noise_patterns = [
            "logo", "icon", "sprite", "banner", "ads", "spacer", "pixel",
            "loader", "placeholder", "social", "button", "nav", "menu"
        ]

        cloudinary.config(
            cloud_name=os.getenv("CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET_KEY")
        )

    def upload_image_to_cloudinary(self,image_bytes):
        try:
            result = cloudinary.uploader.upload(
                image_bytes,
                resource_type="image"
            )
            return result["secure_url"]
        except Exception as e:
            print("Upload failed:", e)
            return None

    def extract_images_with_metadata(self, html, base_url=None):
        if not html: return []
        soup = BeautifulSoup(html, "html.parser")
        found_images = {}

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src: continue

            src = urljoin(base_url, src) if base_url else src

            alt = img.get("alt", "").strip()

            if any(p in src.lower() for p in self.noise_patterns): continue

            score = 0
            if len(alt) > 10: score += 10
            if "chart" in alt.lower() or "table" in alt.lower(): score += 20

            found_images[src] = {"url": src, "alt": alt, "relevance": score}

        sorted_imgs = sorted(found_images.values(), key=lambda x: x['relevance'], reverse=True)
        return sorted_imgs

    async def describe_image_with_vlm(self, image_bytes):
        """
        Executes Gemini 1.5 Flash to extract text and data.
        """
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            prompt = """
                    ACT AS: Professional Document Digitizer & OCR Specialist.
                    TASK: Extract all textual and data-driven content from the provided image.

                    RULES:
                    1. TABLES: Convert all tables into clean GitHub-Flavored Markdown.
                    2. CHARTS: Describe the axes, data points, and the overall trend.
                    3. TEXT: Extract all text preserving paragraph structure and headings (#, ##).
                    4. LABELS: If this is a diagram, list all labeled components.
                    5. IGNORE: Do not describe colors, background textures, or artistic elements.

                    OUTPUT: Provide raw text/markdown only. Do not add "Here is the extraction" or preamble.
                    """

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            )

            response = await self.vlm.ainvoke([message])
            return response.content.strip()

        except Exception as e:
            logging.error(f"Gemini VLM Analysis Failed: {e}")
            return ""



    async def process_all_images(self, html_content, url, max_to_process=3):
        """
        Extract images → convert to bytes → OCR → fallback VLM → return clean text
        """

        image_descriptions = []
        images = self.extract_images_with_metadata(html_content, base_url=url)

        processed_count = 0

        for img_obj in images:
            if processed_count >= max_to_process:
                break

            img_url = img_obj['url']
            alt_text = img_obj['alt']

            try:
                response = requests.get(img_url, timeout=5)
                if response.status_code != 200:
                    continue
                image_bytes = response.content
                print(f"Processing image {processed_count + 1} with Gemini VLM...")
                extracted_text = await self.describe_image_with_vlm(image_bytes)

                if not extracted_text or "NO_TEXT_FOUND" in extracted_text:
                    continue

                entry = f"--- Image Content from {img_url} ---\n{extracted_text}"
                if alt_text and alt_text.lower() not in extracted_text.lower():
                    entry += f"\n[Context/Alt: {alt_text}]"

                image_descriptions.append(entry)
                processed_count += 1

            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue

        return image_descriptions


