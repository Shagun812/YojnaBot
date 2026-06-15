import os, time, json, requests
from adaption import Adaption, DatasetTimeout
from dotenv import load_dotenv
load_dotenv()

client = Adaption(api_key=os.getenv("ADAPTION_API_KEY"))
STATE_FILE = "data/adaption_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(data: dict):
    state = load_state()
    state.update(data)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def initial_ingest():
    """Run this ONCE to seed adapted_eligibility_final.jsonl into Adaption."""
    result = client.datasets.upload_file(
        "data/adapted_eligibility_final.jsonl",
        name="yojnabot-schemes-v1"
    )
    dataset_id = result.dataset_id
    result = client.datasets.upload_file(
    "data/stripped_schemes.jsonl",
    name="yojnabot-schemes-v1"
)

    while True:
        status = client.datasets.get_status(dataset_id)
        if status.row_count is not None:
            print(f"Rows ready: {status.row_count}")
            break
        time.sleep(2)
        run = client.datasets.run(
    dataset_id,
    run = client.datasets.run(
    dataset_id,
    column_mapping={
        "prompt": "instruction",
        "completion": "response"
    }
)
)
    print(f"Adaptation started. Est {run.estimated_minutes} min, {run.estimated_credits_consumed} credits")

    final = client.datasets.wait_for_completion(dataset_id, timeout=1800)
    print(f"Done: {final.status}")
    if hasattr(final, 'error') and final.error:
        print(f"Error details: {final.error}")
    score = get_eval_score(dataset_id)
    save_state({"dataset_id": dataset_id, "eval_score": score})
    print(f"Eval score: {score}")
    return dataset_id

def get_eval_score(dataset_id=None):
    try:
        if not dataset_id:
            dataset_id = load_state().get("dataset_id")
        if not dataset_id:
            return None
        ev = client.datasets.get_evaluation(dataset_id)
        if hasattr(ev, "metrics") and ev.metrics:
            return round(float(list(ev.metrics.values())[0]), 3)
    except Exception as e:
        print(f"Eval error: {e}")
    return None

def push_correction_and_readapt(scheme_name: str, correction_text: str):
    """Called when user submits a correction. Triggers live re-adaptation."""
    entry = {
        "instruction": f"Correct information about {scheme_name}",
        "response": correction_text
    }
    with open("data/corrections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    result = client.datasets.upload_file(
        "data/corrections.jsonl",
        name="yojnabot-corrections"
    )
    cid = result.dataset_id
    while True:
        if client.datasets.get_status(cid).row_count is not None:
            break
        time.sleep(2)

    client.datasets.run(
        cid,
        column_mapping={"prompt": "instruction", "completion": "response"}
    )
    client.datasets.wait_for_completion(cid, timeout=1800)
    new_score = get_eval_score(cid)
    save_state({"dataset_id": cid, "eval_score": new_score})
    return new_score

if __name__ == "__main__":
    initial_ingest()