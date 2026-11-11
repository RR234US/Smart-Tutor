import os
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =====================================================================
# 1. ADAPTIVE TEACHING STYLE SYSTEM PROMPTS
# =====================================================================
STYLE_PROMPTS: Dict[str, str] = {
    "ELI10 (Real-World Analogy)": (
        "You are a friendly AI tutor explaining concepts to a 10-year-old beginner. "
        "Use a fun, everyday real-world analogy (like a kitchen, traffic, or school) to explain "
        "the concept based ONLY on the provided context. Keep sentences simple and jargon-free."
    ),
    "Exam Topper (10-Mark Structured Notes)": (
        "You are a university B.Tech exam coach. Using ONLY the provided context, format your "
        "answer for maximum exam marks: start with a 1-sentence formal Definition, followed by "
        "numbered Key Points, and end with an 'Exam Takeaway' summary."
    ),
    "Socratic Professor (Intuition + Quiz)": (
        "You are a Socratic engineering professor. Explain the core intuition clearly using ONLY "
        "the provided context, show how cause and effect work, and end your response with one "
        "thought-provoking conceptual question for the student to think about."
    ),
    "Practical Engineer (System & Code Focus)": (
        "You are a Senior Systems Architect. Explain the concept using ONLY the provided context "
        "by focusing on system performance, memory/CPU trade-offs, and a short pseudocode or "
        "practical system flow example."
    ),
}


class OpenSourceRAGTutor:
    def __init__(self, notes_path: str = "data/subject_notes.md"):
        self.notes_path = notes_path
        # 1. Load 100% Open-Source Embedding Model (all-MiniLM-L6-v2 -> 384-D vectors)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Load 100% Open-Source Instruction LLM (Qwen2.5-0.5B-Instruct runs fast on CPU)
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # 3. Initialize In-Memory ChromaDB Vector Store with Cosine Similarity
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="subject_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )
        self.subject_title = "Custom Subject"
        self.chunk_count = 0
        self._ingest_knowledge_base()

    def _chunk_markdown_by_section(self, raw_text: str) -> List[Dict[str, str]]:
        """Splits the markdown notes by '##' headings so each topic stays intact."""
        lines = raw_text.splitlines()
        chunks: List[Dict[str, str]] = []
        current_title = "Overview"
        current_body: List[str] = []

        for line in lines:
            if line.startswith("# "):
                self.subject_title = line.replace("# ", "").strip()
            elif line.startswith("## "):
                if current_body:
                    chunks.append({
                        "topic": current_title,
                        "content": "\n".join(current_body).strip(),
                    })
                current_title = line.replace("## ", "").strip()
                current_body = [line]
            else:
                if line.strip():
                    current_body.append(line)

        if current_body:
            chunks.append({
                "topic": current_title,
                "content": "\n".join(current_body).strip(),
            })
        return chunks

    def _ingest_knowledge_base(self) -> None:
        """Reads data/subject_notes.md, embeds chunks, and stores them in ChromaDB."""
        if not os.path.exists(self.notes_path):
            raise FileNotFoundError(f"Missing knowledge base file at: {self.notes_path}")

        with open(self.notes_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = self._chunk_markdown_by_section(raw_text)
        self.chunk_count = len(chunks)

        # Clear existing items if reloading
        existing = self.collection.get()
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [{"topic": c["topic"]} for c in chunks]
        embeddings = self.embedder.encode(documents).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def retrieve_context(self, query: str, top_k: int = 2) -> List[Dict[str, object]]:
        """Retrieves Top-K chunks and computes Cosine Similarity = 1 - Cosine Distance."""
        query_vec = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=min(top_k, self.chunk_count),
        )

        retrieved: List[Dict[str, object]] = []
        for i in range(len(results["documents"][0])):
            cos_distance = float(results["distances"][0][i])
            cos_similarity = round(1.0 - cos_distance, 4)
            retrieved.append({
                "topic": results["metadatas"][0][i]["topic"],
                "content": results["documents"][0][i],
                "similarity": cos_similarity,
            })
        return retrieved

    def generate_styled_answer(
        self, query: str, style_name: str, top_k: int = 2
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Retrieves context via RAG and generates a style-adapted response."""
        retrieved_chunks = self.retrieve_context(query, top_k=top_k)
        context_block = "\n\n---\n\n".join(
            [f"[{c['topic']}]\n{c['content']}" for c in retrieved_chunks]
        )

        style_instruction = STYLE_PROMPTS.get(
            style_name, STYLE_PROMPTS["Exam Topper (10-Mark Structured Notes)"]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{style_instruction}\n\n"
                    "STRICT GROUNDING RULE: Answer using ONLY the Reference Context below. "
                    "If the answer is not in the Reference Context, state clearly: "
                    "'This topic is not covered in the current subject notes.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference Context:\n{context_block}\n\n"
                    f"Student Question: {query}\n\n"
                    f"Provide your explanation in the requested '{style_name}' style:"
                ),
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt")

        with torch.no_grad():
            output_ids = self.llm.generate(
                **inputs,
                max_new_tokens=280,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer, retrieved_chunks