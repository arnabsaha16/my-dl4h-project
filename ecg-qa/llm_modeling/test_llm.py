import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

def query_llm(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Change this to either Llama or Gemma
    #model_name = "meta-llama/Llama-3.1-8B"   # or "google/gemma-2-2b"
    #tokenizer, model = init_llm(model_name)

    #prompt = "Question: Does this ECG show atrial fibrillation?\nOptions: atrial fibrillation, myocardial infarction, none\nAnswer:"
    #answer = query_llm(prompt, tokenizer, model)

    #print("Model output:", answer)
    parser = argparse.ArgumentParser(description="Run a test prompt on Llama or Gemma")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model name, e.g. meta-llama/Llama-3.1-8B or google/gemma-2-2b")
    args = parser.parse_args()

    tokenizer, model = init_llm(args.model_name)

    prompt = "Question: Does this ECG show atrial fibrillation?\nOptions: atrial fibrillation, myocardial infarction, none\nAnswer:"
    answer = query_llm(prompt, tokenizer, model)

    print(f"Model used: {args.model_name}")
    print("Model output:", answer)
