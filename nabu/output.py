import json


def load_output(output: str):
    try:
        return json.loads(output)
    except Exception:
        print(f"Failed to load output {output}")
        return None