# src/llm_handler.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LocalLLMHandler:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=128, device=None):
        # Detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    def generate_response(self, context, question):
        """
        Simple prompt formatting: Adds a header, context, and user question.
        """
        prompt = (
            "You are a helpful assistant. Use the information provided to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "ANSWER:"
        )
        outputs = self.generator(prompt, max_new_tokens=128, clean_up_tokenization_spaces=True)
        answer = outputs[0]['generated_text'][len(prompt):].strip()
        return answer

# --- Example usage ---
if __name__ == "__main__":
    # Example dummy context and question
    dummy_context = (
        "Loan Application ID: LP001002\n"
        "Gender: Male\nMarried: Yes\nLoan_Status: Y\nApplicantIncome: 5000\n"
    )
    user_question = "Was this loan application approved, and what was the applicant's income?"

    llm = LocalLLMHandler()
    answer = llm.generate_response(dummy_context, user_question)
    print("LLM Answer:\n", answer)
