import os
import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from openai import OpenAI
import cloudinary
import cloudinary.uploader
import easyocr
import numpy as np
from PIL import Image
import io
import requests

class ImageAnalyzer:
    """
    A professional-grade image extraction and analysis class for RAG pipelines.
    Combines heuristic filtering with VLM (Vision LLM) context-aware captioning.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("QWEN_API_KEY")
        )
        self.model = "qwen/qwen-2.5-vl-7b-instruct"
        self.noise_patterns = [
            "logo", "icon", "sprite", "banner", "ads", "spacer", "pixel",
            "loader", "placeholder", "social", "button", "nav", "menu"
        ]

        cloudinary.config(
            cloud_name=os.getenv("CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET_KEY")
        )
        self.reader = easyocr.Reader(['en'])

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
        """
        Extracts high-quality images and their associated Alt-text.
        Filters out UI noise to ensure only relevant content is analyzed.
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        found_images = {}

        for img in soup.find_all("img"):
            src = (
                    img.get("data-src") or
                    img.get("data-lazy-src") or
                    img.get("data-original") or
                    img.get("src")
            )
            alt = img.get("alt", "").strip()

            srcset = img.get("srcset")
            if srcset:
                try:
                    src = srcset.split(",")[-1].strip().split(" ")[0]
                except:
                    pass

            if not src:
                continue

            if src.startswith("//"):
                src = "https:" + src
            if base_url:
                src = urljoin(base_url, src)

            src_lower = src.lower()

            if any(ext in src_lower for ext in [".svg", ".gif", ".ico"]) or src.startswith("data:"):
                continue
            if any(pattern in src_lower for pattern in self.noise_patterns):
                continue

            try:
                width = img.get("width")
                height = img.get("height")
                if width and height:
                    w = int(str(width).replace('px', ''))
                    h = int(str(height).replace('px', ''))
                    if w < 150 or h < 150:
                        continue
            except (ValueError, TypeError):
                pass

            if src not in found_images or (len(alt) > len(found_images[src]['alt'])):
                found_images[src] = {"url": src, "alt": alt}

        return list(found_images.values())

    def describe_image_with_vlm(self, image_bytes):
        """
        Generates a technical description using a Vision LLM.
        Handles both standard API objects and raw error strings from OpenRouter.
        """
        try:
            image_url = self.upload_image_to_cloudinary(image_bytes)

            prompt = """
            You are an OCR + data extraction system.

            STRICT RULES:
            - Extract ONLY what is explicitly visible
            - DO NOT describe the image
            - DO NOT summarize
            - DO NOT infer relationships
            - DO NOT guess missing values

            TASK:
            Return ALL visible text, numbers, labels, and structured content exactly as seen.

            FORMAT:
            - Preserve line breaks
            - Preserve tables (row-wise if possible)
            - Preserve headings and sections
            - Keep original wording

            If no text is visible, return:
            NO_TEXT_FOUND
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                timeout=20
            )

            if isinstance(response, str):
                logging.error(f"OpenRouter returned error string: {response}")
                return ""


            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()

            return ""

        except Exception as e:
            logging.error(f"VLM Analysis Failed for {image_url}: {e}")
            return ""



    def process_all_images(self, html_content, url, max_to_process=3):
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

                extracted_text, confidence = self.extract_text_from_image_bytes(image_bytes)

                print("\n[OCR OUTPUT]", extracted_text)
                print("[OCR CONFIDENCE]", confidence)

                if len(extracted_text.strip()) < 20 or confidence < 0.6:
                    print(" OCR weak → using VLM fallback")

                    extracted_text = self.describe_image_with_vlm(image_bytes)

                    print("\n[VLM OUTPUT]", extracted_text)

                if not extracted_text or len(extracted_text.strip()) < 10:
                    continue

                cleaned_text = self.clean_ocr_text(extracted_text)

                entry = f"Image content from {url}:\n{cleaned_text}"

                if alt_text and alt_text.lower() not in cleaned_text.lower():
                    entry += f"\n[Context: {alt_text}]"

                image_descriptions.append(entry)
                processed_count += 1

            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue

        return image_descriptions

    def extract_text_from_image_bytes(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)

            results = self.reader.readtext(image_np)

            lines = []
            total_conf = 0

            for (_, text, prob) in results:
                if prob > 0.5:
                    lines.append(text)
                    total_conf += prob

            avg_conf = total_conf / max(len(lines), 1)

            extracted = "\n".join(lines)

            return extracted, avg_conf

        except Exception as e:
            print(f"OCR failed: {e}")
            return "", 0

    def clean_ocr_text(self, text):
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        seen = set()
        unique_lines = []

        for line in lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

        return "\n".join(unique_lines)

