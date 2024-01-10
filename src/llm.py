import json
import fire
from vllm import LLM, SamplingParams


class DataManager():

    def __init__(self, data_path: str, save_path: str, key: str):
        self.data_path = data_path
        self.save_path = save_path
        self.key = key
        self.templates = {"mistral": "<s> [INST] {instruction} [/INST]",
                          "llama": "<s> [INST] {instruction} [/INST]"}
        
    def set_prompt(self, model_path: str):
        self.prompt = ""
        for model_key in self.templates:
            if model_key in model_path.lower():
                self.prompt = self.templates[model_key]
    
    def load_data(self):
        data = list(map(json.loads, open(self.data_path)))
        prompts = [self.prompt.format(instruction=x[self.key]) for x in data]
        
        return prompts

    def save_data(self, prompts, responses):
        with open(self.save_path, "w") as file:
            for prompt, response in zip(prompts, responses):
                instance = \
                {
                    "prompt": prompt,
                    "output": response.outputs[0].text.strip(),
                    "logprob": response.outputs[0].cumulative_logprob,
                    "prompt_logprobs": response.prompt_logprobs,
                }
                print(json.dumps(instance), file=file)


class Pipeline:

    def __init__(self, model_path, temperature, top_p, max_tokens, dtype, quantization):
        self.llm = LLM(model=model_path,
                       quantization=quantization,
                       dtype=dtype,)
        self.sampling_params = SamplingParams(temperature=temperature,
                                              top_p=top_p,
                                              max_tokens=max_tokens,
                                              prompt_logprobs=0)

    def generate(self, prompts):
        responses = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return responses


def run(data_path: str,
        output_file: str,
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 128,
        key: str = "sentence",
        dtype: str = "bfloat16",
        quantization: str = None):
    
    data_manager = DataManager(data_path, output_file, key)
    data_manager.set_prompt(model_path)
    prompts = data_manager.load_data()
    pipeline = Pipeline(model_path, temperature, top_p, max_tokens, dtype, quantization)

    responses = pipeline.generate(prompts)
    data_manager.save_data(prompts, responses)


if __name__ == '__main__':
    fire.Fire(run)