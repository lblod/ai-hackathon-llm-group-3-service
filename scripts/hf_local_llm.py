import torch
from transformers import pipeline

class LocalHFLLM:
    def __init__(self, model_name="BramVanroy/fietje-2-instruct",
                 max_new_tokens=256,
                 do_sample=False,
                 temperature=0.7,
                 top_k=50,
                 top_p=0.95):
        self.model = pipeline("text-generation",
                              model=model_name,
                              torch_dtype=torch.bfloat16,
                              device_map="auto")
        self.max_new_tokens = max_new_tokens

    def run(self, messages):
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.do_sample:
            outputs = model(prompt,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True,
                            temperature=self.temperature,
                            top_k=self.top_k,
                            top_p=self.top_p)
        else:
            outputs = self.model(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        full_answer = outputs[0]["generated_text"]
        return full_answer

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
        full_answer = self.run(messages)
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
        full_answer = self.run(messages)
        answer = full_answer.split('<|assistant|>')[1]
        return answer
