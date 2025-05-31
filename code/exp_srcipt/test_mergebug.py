import json

def extract_keys_to_set():
        with open('24hour.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        key_set = set(data.keys())
        print(f"Total unique keys: {len(key_set)}")

        return key_set

extract_keys_to_set()

