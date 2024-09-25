import torch
from transformers import pipeline

class LocalHFLLM:
    def __init__(self, model_name):
        self.model = pipeline("text-generation",
                              model=model_name,
                              torch_dtype=torch.bfloat16,
                              device_map="auto")

    def request(system_prompt, user_prompt):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        outputs = self.model(prompt, max_new_tokens=256, do_sample=False)
        full_answer = outputs[0]["generated_text"]
        answer = full_answer.split('<|assistant|>')[1]
        return answer

    def request_with_context(system_prompt, user_prompt, context):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"{user_prompt}\ncontext:{context}"
            }
        ]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        outputs = self.model(prompt, max_new_tokens=256, do_sample=False)
        full_answer = outputs[0]["generated_text"]
        answer = full_answer.split('<|assistant|>')[1]
        return answer
