from langchain_community.llms import Ollama
from rag_project.rag.prompts.prompt_templates import RAG_PROMPT


class OllamaLLM:
    """
    Ollama LLM wrapper (phi).
    Loads model once and reuses it.
    """

    def __init__(self, model_name: str = "phi"):
        print("\n[LLM] Initializing Ollama...")
        self.llm = Ollama(model=model_name)
        print("[LLM] Ollama ready\n")

    def generate(self, question: str, context: str) -> str:
        prompt = RAG_PROMPT.format(
            question=question,
            context=context
        )
        return self.llm.invoke(prompt)
