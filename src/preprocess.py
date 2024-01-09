import fire
import json


def main(data_path: str = "data/ptb.json",
         output_file: str = "data/ptb.jsonl"):
    
    data = []
    
    with open(data_path) as file:
        instances = list(map(json.loads, file))
        rows = instances[0]["rows"]
        for row in rows:
            data.append({"sentence": row["row"]["sentence"]})

    with open(output_file, "w") as file:
        for ins in data:
            print(json.dumps(ins), file=file)


if __name__ == "__main__":
    fire.Fire(main)
