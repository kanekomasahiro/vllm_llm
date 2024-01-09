import json
import fire
from vllm import LLM, SamplingParams


PROMPTS = {
    "mistral": "<s> [INST] {instruction} [/INST]",
    "llama": "<s> [INST] {instruction} [/INST]"
}

def run(data_path: str,
        output_file: str,
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 128,
        key: str = "sentence",
        quantization: str = None):
    
    llm = LLM(
        model=model_path,
        quantization=quantization,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    prompt = ""
    for model_key in PROMPTS:
        if model_key in model_path.lower():
            prompt = PROMPTS[model_key]
    assert prompt

    data = list(map(json.loads, open(data_path)))
    prompts = [prompt.format(instruction=x[key]) for x in data]
    responses = llm.generate(prompts, sampling_params=sampling_params)

    with open(output_file, "w") as file:
        for ins, response in zip(data, responses):
            ins.update({
                "output": response.outputs[0].text.strip(),
                "logprob": response.outputs[0].cumulative_logprob
            })
            print(json.dumps(ins), file=file)

if __name__ == '__main__':
    fire.Fire(run)