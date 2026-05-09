"""
RAG Pipeline Module
===================
Core Retrieval-Augmented Generation pipeline. When triggered by the
severity analyzer, it:
  1. Builds a context-aware query from the fall event data
  2. Retrieves relevant triage/radiology documents from ChromaDB
  3. Synthesizes a structured triage brief (via LLM or template fallback)

Uses TF-IDF embeddings (fully offline, no API key needed for retrieval).
Set OPENAI_API_KEY env var for LLM-enhanced synthesis (optional).
"""

import os
import json
import pickle
import chromadb
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# ── Configuration ───────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "chroma_db", "tfidf_model.pkl")
COLLECTION_NAME = "triage_knowledge"
TOP_K = 6
EMBEDDING_DIM = 384

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


@dataclass
class TriageBrief:
    alert_id: str
    timestamp: str
    severity: str
    triage_color: str
    fall_type: str
    impact_type: str
    immediate_actions: List[str]
    injury_risk_assessment: Dict[str, List[str]]
    recommended_imaging: List[Dict]
    transport_priority: str
    transport_notes: str
    sources: List[str]
    retrieved_context: List[Dict]
    llm_used: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _load_tfidf_vectorizer():
    """Load the fitted TF-IDF vectorizer from disk."""
    with open(TFIDF_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _embed_query(query_text, vectorizer):
    """Embed a query string using the fitted TF-IDF vectorizer."""
    matrix = vectorizer.transform([query_text])
    emb = matrix.toarray()[0].tolist()
    if len(emb) < EMBEDDING_DIM:
        emb += [0.0] * (EMBEDDING_DIM - len(emb))
    return [emb]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for fall triage support."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.vectorizer = _load_tfidf_vectorizer()
        self._llm_available = bool(OPENAI_API_KEY)

    def build_query(self, severity_result) -> str:
        return (
            f"Fall event: {severity_result.fall_type}, "
            f"severity {severity_result.severity}, "
            f"{severity_result.impact_type}. "
            f"Peak lateral acceleration: {severity_result.lateral_g}g. "
            f"Impact direction: {severity_result.impact_direction}. "
            f"Angular velocity: {severity_result.rotation_speed} rad/s. "
            f"Heart rate change: {severity_result.heart_rate_delta} bpm. "
            f"Final position: {severity_result.final_orientation}. "
            f"Risk factors: {', '.join(severity_result.risk_factors)}. "
            f"Need: triage protocol, radiological assessment guidelines, "
            f"immediate response actions."
        )

    def retrieve(self, query: str, n_results: int = TOP_K) -> List[Dict]:
        query_embedding = _embed_query(query, self.vectorizer)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "category": results["metadatas"][0][i].get("category", "unknown"),
                "filename": results["metadatas"][0][i].get("filename", "unknown"),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return retrieved

    def _synthesize_with_llm(self, query, retrieved_chunks, severity_result):
        from openai import OpenAI
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}]\n{c['content']}" for c in retrieved_chunks
        ])
        prompt = f"""You are a medical triage assistant for an automated fall detection system.
Based on the fall event data and retrieved clinical guidelines below, produce a structured triage brief.

## Retrieved Clinical Guidelines:
{context}

## Fall Event Data:
- Fall type: {severity_result.fall_type}
- Severity: {severity_result.severity}
- Impact type: {severity_result.impact_type}
- Peak lateral G-force: {severity_result.lateral_g}g
- Total G-force: {severity_result.total_g}g
- Impact direction: {severity_result.impact_direction}
- Angular velocity at impact: {severity_result.rotation_speed} rad/s
- Heart rate change: {severity_result.heart_rate_delta} bpm
- Final orientation: {severity_result.final_orientation}
- Risk factors: {', '.join(severity_result.risk_factors)}

## Return ONLY valid JSON:
{{"immediate_actions":["..."],"injury_risk_high":["..."],"injury_risk_moderate":["..."],"injury_risk_low":["..."],"imaging":[{{"type":"...","region":"...","urgency":"STAT/URGENT/ROUTINE","rationale":"..."}}],"transport_priority":"IMMEDIATE/URGENT/ROUTINE","transport_notes":"...","triage_color":"RED/YELLOW/GREEN"}}"""

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1500,
        )
        try:
            raw = response.choices[0].message.content
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            return self._synthesize_template(query, retrieved_chunks, severity_result)

        sources = list(set(c["source"] for c in retrieved_chunks))
        return TriageBrief(
            alert_id=f"FALL-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=severity_result.severity,
            triage_color=data.get("triage_color", "RED"),
            fall_type=severity_result.fall_type,
            impact_type=severity_result.impact_type,
            immediate_actions=data.get("immediate_actions", []),
            injury_risk_assessment={
                "high_probability": data.get("injury_risk_high", []),
                "moderate_probability": data.get("injury_risk_moderate", []),
                "low_probability": data.get("injury_risk_low", []),
            },
            recommended_imaging=data.get("imaging", []),
            transport_priority=data.get("transport_priority", "IMMEDIATE"),
            transport_notes=data.get("transport_notes", ""),
            sources=sources, retrieved_context=retrieved_chunks, llm_used=True,
        )

    def _synthesize_template(self, query, retrieved_chunks, severity_result):
        """Template-based fallback — no LLM needed."""
        sources = list(set(c["source"] for c in retrieved_chunks))

        immediate_actions = [
            "Do NOT move the subject — assess for spinal injury first",
            "Check consciousness using AVPU scale (Alert/Voice/Pain/Unresponsive)",
            "Assess circulation: radial pulse, capillary refill time",
            "Monitor vital signs continuously",
        ]
        injury_high, injury_moderate, injury_low = [], [], []
        imaging = []

        if severity_result.impact_direction in ("left", "right"):
            immediate_actions.append("Immobilize the affected limb in the position found — do NOT apply traction")
            injury_high.append(f"Hip fracture ({severity_result.impact_direction} lateral impact, {severity_result.lateral_g}g)")
            imaging.append({"type": "X-Ray", "region": "Pelvis AP + Lateral Hip", "urgency": "STAT",
                            "rationale": f"High-energy lateral fall ({severity_result.lateral_g}g), hip fracture likely"})

        if severity_result.rotation_speed > 3.0:
            immediate_actions.append("Apply cervical collar if head/neck impact suspected")
            injury_high.append(f"Head injury — uncontrolled rotation ({severity_result.rotation_speed:.1f} rad/s)")
            imaging.append({"type": "CT", "region": "Head Non-contrast", "urgency": "URGENT",
                            "rationale": "Canadian CT Head Rule: fall with high angular velocity"})

        if severity_result.total_g > 4.0:
            injury_moderate.append("Rib fractures (high total G-force impact)")
            imaging.append({"type": "X-Ray", "region": "Chest PA/Lateral", "urgency": "URGENT",
                            "rationale": f"High-energy impact ({severity_result.total_g}g), rule out rib fractures"})

        injury_moderate.append("Wrist fracture (FOOSH — Fall On Outstretched Hand)")
        injury_low.append("Subdural hematoma (if anticoagulated)")
        injury_low.append("Pneumothorax (if rib fractures present)")

        if severity_result.heart_rate_delta > 20:
            immediate_actions.append(
                f"Heart rate spike detected (+{severity_result.heart_rate_delta:.0f} bpm) — assess for pain/shock"
            )

        if severity_result.final_orientation == "lying_on_side":
            immediate_actions.append("Subject in lateral position — assess spinal tenderness before any movement")
            imaging.append({"type": "CT", "region": "Cervical Spine", "urgency": "URGENT",
                            "rationale": "Lateral fall with rotational component — NEXUS criteria likely not clearable"})

        triage_color = "RED" if severity_result.severity == "HIGH" else "YELLOW"
        transport_priority = "IMMEDIATE" if severity_result.severity == "HIGH" else "URGENT"
        transport_notes = ("Spinal precautions required. Use scoop stretcher. "
                           "Log-roll technique if repositioning needed (min 4 persons). "
                           "Transport to facility with orthopedic surgery and CT capability."
                           if severity_result.severity == "HIGH" else "Standard stretcher transport. Monitor en route.")

        return TriageBrief(
            alert_id=f"FALL-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=severity_result.severity,
            triage_color=triage_color,
            fall_type=severity_result.fall_type,
            impact_type=severity_result.impact_type,
            immediate_actions=immediate_actions,
            injury_risk_assessment={
                "high_probability": injury_high,
                "moderate_probability": injury_moderate,
                "low_probability": injury_low,
            },
            recommended_imaging=imaging,
            transport_priority=transport_priority,
            transport_notes=transport_notes,
            sources=sources, retrieved_context=retrieved_chunks, llm_used=False,
        )

    def generate_triage_brief(self, severity_result):
        query = self.build_query(severity_result)
        retrieved_chunks = self.retrieve(query)

        if self._llm_available:
            try:
                return self._synthesize_with_llm(query, retrieved_chunks, severity_result)
            except Exception as e:
                print(f"⚠️  LLM failed ({e}), using template fallback.")
                return self._synthesize_template(query, retrieved_chunks, severity_result)
        else:
            return self._synthesize_template(query, retrieved_chunks, severity_result)


# ── Convenience functions ───────────────────────────────────────────────
_pipeline_instance: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance

def run_triage(severity_result) -> Optional[TriageBrief]:
    if not severity_result.trigger_rag:
        return None
    pipeline = get_pipeline()
    return pipeline.generate_triage_brief(severity_result)
