import json


def load_output(output: str):
    try:
        response = json.loads(output)
    except Exception as e:
        print(f"Failed to load output {output}")
    return response