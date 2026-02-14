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
        with open(project_root / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

        llm_config = config["llm"]

        self.max_context_chars = llm_config.get("max_context_chars", 4000)
        self.context_truncate = llm_config.get("context_truncate", True)

        self.llm = Ollama(
            model=llm_config["model"],
            temperature=llm_config.get("temperature", 0.2),
            num_predict=llm_config.get("max_tokens", 256),
            top_p=llm_config.get("top_p", 0.9),
            repeat_penalty=llm_config.get("repeat_penalty", 1.1),
        )

        print("[LLM] Ollama ready\n")

    def generate(self, question: str, context: str) -> str:
        """
        Generate grounded answer using RAG prompt.
        """

        # Optional context truncation (controlled by config)
        if self.context_truncate and len(context) > self.max_context_chars:
            context = context[:self.max_context_chars]

        prompt = RAG_PROMPT.format(
            question=question,
            context=context
        )

        return self.llm.invoke(prompt)
