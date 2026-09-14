"""
Tier 3 (L3: Refinement) LLM Provider Abstraction and Deterministic Mock Implementation.

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Maps to paper methodology: using LLMs to synthesize, clean, and convert selected
web text into high-density educational formats (Q&A pairs, textbook explanations).
Deterministic offline implementation avoids API keys, GPU hardware, and network calls.
All mock outputs are explicitly tagged with 'mock_llm'.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM synthesis and refinement providers."""

    @abstractmethod
    def refine_document(self, text: str, doc_id: str = "") -> dict[str, Any]:
        """
        Synthesize and refine document into educational formats.

        Returns:
            Dict containing:
            - text_refined (str): Polished, structured educational text
            - qa_pair (dict[str, str]): Generated question and answer
            - textbook_explanation (str): Structured chapter-style explanation
            - generated_by (str): Tag identifying the synthesizer
        """
        pass


class MockLLMProvider(LLMProvider):
    """
    Offline deterministic mock synthesizer.

    Performs rule-based extraction and structuring to emulate LLM refinement
    without requiring GPUs, weights, or external API access.
    All outputs are explicitly tagged with 'mock_llm'.
    """

    def __init__(self, model_name: str = "mock-rule-synthesizer-v1") -> None:
        self.model_name = model_name

    def _extract_primary_topic(self, text: str) -> str:
        """Extract the likely primary subject/noun phrase from the opening sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        first_sentence = sentences[0] if sentences else text
        
        # Look for subject before common verbs
        match = re.match(r"^([A-Z][\w\s\-]+?)\s+(?:is|was|are|were|introduced|demonstrates|describes|provides|refers|focuses)", first_sentence)
        if match:
            topic = match.group(1).strip()
            if len(topic.split()) <= 6:
                return topic

        # Fallback to first few words
        words = first_sentence.split()
        return " ".join(words[:4]).strip(" ,.:;") if words else "Machine Learning Concept"

    def refine_document(self, text: str, doc_id: str = "") -> dict[str, Any]:
        """
        Deterministically transform input text into:
        1. Cleaned / refined text
        2. Question-Answer pair
        3. Textbook-style explanation
        """
        topic = self._extract_primary_topic(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        
        # 1. Refined text: polished paragraphs with structural header
        refined_paragraphs = [
            f"### Overview: {topic}",
            " ".join(sentences[:2]) if len(sentences) >= 2 else text,
        ]
        if len(sentences) > 2:
            refined_paragraphs.append(" ".join(sentences[2:]))
        text_refined = "\n\n".join(refined_paragraphs)

        # 2. Q&A pair: synthesized conceptual question and grounded answer
        question = f"What is the significance of {topic}, and how does it operate?"
        if len(sentences) >= 2:
            answer = f"{sentences[0]} In practice, {sentences[1].lower() if sentences[1][0].isupper() else sentences[1]}"
        else:
            answer = text

        qa_pair = {
            "question": question,
            "answer": answer,
        }

        # 3. Textbook explanation: structured chapter layout
        key_takeaways = "\n".join([f"- {s}" for s in sentences[:3]])
        textbook_explanation = (
            f"# Chapter: {topic}\n\n"
            f"## 1. Introduction and Core Definition\n"
            f"{sentences[0] if sentences else text}\n\n"
            f"## 2. Key Principles and Mechanism\n"
            f"{' '.join(sentences[1:]) if len(sentences) > 1 else 'Further architectural nuances build upon these foundational principles.'}\n\n"
            f"## 3. Conceptual Summary\n"
            f"{key_takeaways}\n"
        )

        return {
            "text_refined": text_refined,
            "qa_pair": qa_pair,
            "textbook_explanation": textbook_explanation,
            "primary_topic": topic,
            "generated_by": "mock_llm",
            "model": self.model_name,
            "validation_status": "pending_validation",
        }


class GeminiLLMProvider(LLMProvider):
    """
    Real LLM synthesis provider leveraging Google Gemini API.

    Extension A for Phase 3 (Tier 3: Refined Knowledge).
    Transforms selected raw web tokens into high-density educational text,
    conceptual Q&A pairs, and structured textbook chapters using a real LLM.

    API and SDK Notes:
    - Targets the `google-generativeai` SDK. Note: Google has introduced
      the newer `google-genai` SDK in late 2024; if `google-generativeai` is deprecated
      in your environment, install `google-genai` and configure accordingly.
    - Model availability: Default is "gemini-1.5-flash". Free-tier model availability
      and names evolve on Google AI Studio (e.g. "gemini-1.5-flash", "gemini-1.5-flash-8b",
      "gemini-2.0-flash"); verify available models in your Google AI Studio console.
    - Fault tolerance: If the API request fails (rate limits, network timeout) or JSON parsing
      fails, this provider falls back gracefully to deterministic MockLLMProvider for that record.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = None) -> None:
        self.model_name = model_name
        self._fallback_provider = MockLLMProvider()

        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not found.\n"
                "To use GeminiLLMProvider (Extension A):\n"
                "1. Obtain a free API key from Google AI Studio: https://aistudio.google.com/\n"
                "2. In Google Colab, add the key in the Secrets tab (key: 'GEMINI_API_KEY') and run:\n"
                "   from google.colab import userdata\n"
                "   import os\n"
                "   os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')\n"
                "3. In local/Linux terminal, export GEMINI_API_KEY='your-key-here'."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)
        except ImportError as err:
            raise ImportError(
                f"google-generativeai package not found ({err}).\n"
                "Please install optional ML dependencies via:\n"
                "    pip install -r requirements-ml.txt\n"
                "or: pip install google-generativeai"
            ) from err

    def refine_document(self, text: str, doc_id: str = "") -> dict[str, Any]:
        """
        Synthesize document using Gemini API with structured JSON output.
        Falls back to MockLLMProvider upon API or parsing failures.
        """
        import json
        import logging

        prompt = (
            "You are an expert educational AI data engineer curating pre-training data for LLMs.\n"
            "Given the following web text, synthesize it into high-density educational formats.\n"
            "Respond ONLY with a valid, strictly formatted JSON object with no preamble or commentary.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "refined_text": "### Overview: <Entity>\\n\\n<Polished, informative 1-2 paragraph summary>",\n'
            '  "question": "A conceptual, educational question testing understanding of the text?",\n'
            '  "answer": "A detailed, grounded, factual answer based directly on the text (at least 15 words).",\n'
            '  "textbook_explanation": "# Chapter: <Topic>\\n\\n## 1. Introduction and Core Definition\\n<Paragraph>\\n\\n## 2. Key Principles and Mechanism\\n<Paragraph>\\n\\n## 3. Conceptual Summary\\n- <Bullet 1>\\n- <Bullet 2>",\n'
            '  "primary_topic": "<Short entity or topic name>"\n'
            "}\n\n"
            f"Input text:\n{text[:3000]}\n"
        )

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "top_p": 0.95},
            )
            raw_output = response.text.strip()
            if raw_output.startswith("```"):
                lines = raw_output.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_output = "\n".join(lines).strip()

            parsed = json.loads(raw_output)

            refined_text = str(parsed.get("refined_text", "")).strip()
            question = str(parsed.get("question", "")).strip()
            answer = str(parsed.get("answer", "")).strip()
            textbook = str(parsed.get("textbook_explanation", "")).strip()
            topic = str(parsed.get("primary_topic", "")).strip() or "Machine Learning Concept"

            if not refined_text or not question or not answer or not textbook:
                raise ValueError("Gemini response was missing required structural fields.")

            return {
                "text_refined": refined_text,
                "qa_pair": {
                    "question": question,
                    "answer": answer,
                },
                "textbook_explanation": textbook,
                "primary_topic": topic,
                "generated_by": "gemini_llm",
                "model": self.model_name,
                "validation_status": "pending_validation",
            }
        except Exception as e:
            logging.warning(
                f"[GeminiLLMProvider] API call or parsing failed for doc {doc_id}: {e}. "
                "Falling back gracefully to MockLLMProvider for this record."
            )
            fallback_res = self._fallback_provider.refine_document(text, doc_id=doc_id)
            fallback_res["generated_by"] = "mock_llm_fallback"
            return fallback_res

