import yaml
from pathlib import Path
from langchain_community.llms import Ollama
from rag_project.rag.prompts.prompt_templates import RAG_PROMPT


class OllamaLLM:
    """
    Ollama LLM wrapper.
    All parameters are controlled via config.yaml.
    """

    def __init__(self, model_name: str = None):
        print("\n[LLM] Initializing Ollama...")

        # Load config
        project_root = Path(__file__).resolve().parents[3]
        with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_config = config["llm"]

        self.max_context_chars = llm_config.get("max_context_chars", 3500)
        self.context_truncate = llm_config.get("context_truncate", True)

        self.llm = Ollama(
            model=llm_config["model"],
            temperature=llm_config.get("temperature", 0.1),
            num_predict=llm_config.get("max_tokens", 128),
            top_p=llm_config.get("top_p", 0.8),
            repeat_penalty=llm_config.get("repeat_penalty", 1.1),
        )

        print("[LLM] Ollama ready\n")

    def _safe_truncate(self, text: str) -> str:
        """
        Truncate context safely without cutting words mid-way.
        """
        if len(text) <= self.max_context_chars:
            return text

        truncated = text[:self.max_context_chars]
        return truncated.rsplit(" ", 1)[0]

    def generate(self, question: str, context: str) -> str:
        """
        Generate grounded answer using strict RAG prompt.
        """

        # Context truncation (controlled via config)
        if self.context_truncate:
            context = self._safe_truncate(context)

        prompt = RAG_PROMPT.format(
            question=question.strip(),
            context=context.strip()
        )

        response = self.llm.invoke(prompt)

        # Ensure clean string output
        if isinstance(response, str):
            return response.strip()

        return str(response).strip()
