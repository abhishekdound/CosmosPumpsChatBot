import os
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
import cloudinary
import cloudinary.uploader

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

    def describe_image_with_vlm(self, image_url, alt_hint=""):
        """
        Generates a technical description using a Vision LLM.
        Handles both standard API objects and raw error strings from OpenRouter.
        """
        try:
            context_msg = f" The image is tagged as '{alt_hint}'." if alt_hint else ""

            prompt = f"""
            You are analyzing an image for a retrieval system.

            {context_msg}

            Your task:

            1. Describe the image clearly in natural language.
            2. Extract all visible factual details.
            3. Capture structure, relationships, and important elements.

            Focus on:
            - objects and components
            - text and labels in the image
            - spatial relationships (above, below, connected to)
            - numbers, measurements, or values (if present)
            - processes or flows (if visible)

            Output format:

            First section: Clear description
            - A short paragraph describing what the image contains

            Second section: Structured facts
            - Write multiple short factual sentences

            Example:
            Description:
            A diagram showing a tank connected to a pump with fluid flowing upward.

            Facts:
            - A tank is located at the left side
            - A pump is connected to the tank
            - Fluid flows from the tank to an elevated outlet
            - The outlet is higher than the tank

            Do NOT hallucinate.
            Do NOT assume missing values.
            Only use visible information.
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
        The main entry point: Extracts images and generates enriched descriptions.
        """
        image_descriptions = []
        images = self.extract_images_with_metadata(html_content, base_url=url)

        processed_count = 0
        for img_obj in images:
            if processed_count >= max_to_process:
                break

            img_url = img_obj['url']
            alt_text = img_obj['alt']

            desc = self.describe_image_with_vlm(img_url, alt_hint=alt_text)

            if desc and len(desc) > 20:
                entry = f"Image at {url}: {desc}"
                if alt_text and alt_text.lower() not in desc.lower():
                    entry += f" (Context: {alt_text})"

                image_descriptions.append(entry)
                processed_count += 1

        return image_descriptions

    def describe_image_bytes(self, image_bytes, alt_hint=""):
        import base64

        try:
            # size check
            if len(image_bytes) > 2_000_000:
                print("Image too large")
                return ""

            image_base64 = base64.b64encode(image_bytes).decode()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract key facts from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }],
                timeout=20
            )

            # 🔥 DEBUG
            print("TYPE:", type(response))

            # 🔥 HANDLE STRING ERROR
            if isinstance(response, str):
                print("OpenRouter ERROR:", response)
                return ""

            # 🔥 SAFE PARSE
            try:
                return response.choices[0].message.content.strip()
            except Exception as e:
                print("Parsing failed:", e)
                return ""

        except Exception as e:
            print("Image failed:", e)
            return ""

