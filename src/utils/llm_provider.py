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
