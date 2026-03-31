import fire
import yaml
import json
import regex as re
import os
from dotenv import load_dotenv

load_dotenv()

from bench_utils import (
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    get_endpoint,
    hash_pil_image,
    load_model_answers,
    load_model_judgements,
    model_name_to_id
)

system_prompt = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below.
...
"""

def get_score(judgement, pattern, pairwise=True):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        return matches[0].strip("\n"), False
    else:
        return None, False


def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    api_dict = get_endpoint(endpoint_dict["endpoints"])
    if endpoint_dict["api_type"] == "anthropic":
        return chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        return chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        return chat_completion_openai(model, conv, temperature, max_tokens, api_dict)


def judgement(**args):
    question = args["question"]
    images = args["images"]
    answer = args["answer"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    output = {
        "question_id": question["question_id"],
        "model": answer["model"],
        "judge": model,
        "games": []
    }

    conv = [{"role": "system", "content": configs["system_prompt"]}]

    for template in configs["prompt_template"]:
        user_prompt = template.format(
            question_1=question["instruction"],
            answer_1=baseline["output"],
            answer_2=answer["output"],
        )

        # ✅ FIX: HASH images BEFORE sending to OpenAI
        conv.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}] + [
                {"type": "image", "image": hash_pil_image(img)} for img in images
            ]
        })

    judgement_text = ""
    try:
        response = get_answer(
            model,
            conv,
            configs["temperature"],
            configs["max_tokens"],
            args["endpoint_dict"]
        )
        judgement_text = response
        score, _ = get_score(judgement_text, args["regex_pattern"])
    except Exception as e:
        judgement_text = f"ERROR: {str(e)}"
        score = None

    output["games"].append({
        "user_prompt": conv[1]["content"],
        "judgement": judgement_text,
        "score": score
    })

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def main(
    judge_config="./config/judge_config.yaml",
    api_config="./config/api_config.yaml",
):
    with open(judge_config) as f:
        configs = yaml.safe_load(f)
    with open(api_config) as f:
        endpoint_list = yaml.safe_load(f)

    pattern = re.compile(configs["regex_pattern"])
    answer_dir = os.path.join("data", configs["bench_name"], "model_answers")

    questions = load_dataset(
        "WildVision/wildvision-arena-data",
        "release_bench_0617_with_modelresponse",
        split="test500"
    )

    model_answers = load_model_answers(answer_dir)
    models = [model_name_to_id(m) for m in configs["model_list"]]

    output_dir = f"data/{configs['bench_name']}/model_judgements/judge_{configs['judge_model']}_reference_{configs['baseline_model']}"
    os.makedirs(output_dir, exist_ok=True)

    endpoint_info = endpoint_list[configs["judge_model"]]
    existing = load_model_judgements(output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            for q in questions:
                qid = q["question_id"]
                if model not in model_answers or qid not in model_answers[model]:
                    continue
                if model in existing and qid in existing[model]:
                    continue

                futures.append(executor.submit(
                    judgement,
                    question=q,
                    images=[q["image"]],
                    answer=model_answers[model][qid],
                    baseline_answer=model_answers[configs["baseline_model"]][qid],
                    configs=configs,
                    endpoint_dict=endpoint_info,
                    output_file=os.path.join(output_dir, f"{model}.jsonl"),
                    regex_pattern=pattern
                ))

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            f.result()


if __name__ == "__main__":
    fire.Fire(main)
