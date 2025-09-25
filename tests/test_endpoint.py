import pandas as pd
import requests
import json

# API_URL = "http://127.0.0.1:8000/predict"
API_URL= "http://housing-app-env.eba-fmh8pgav.us-east-1.elasticbeanstalk.com/predict"

def main():
    # Load unseen data
    df = pd.read_csv("data/future_unseen_examples.csv")

    # Take 3 rows for testing
    examples =  df.sample(3).to_dict(orient="records")

    print(f"Sending {len(examples)} examples to {API_URL}\n")

    for i, example in enumerate(examples, start=1):
        try:
            response = requests.post(API_URL, json=example)

            if response.status_code == 200:
                print(f"Example {i}:")
                print("Input:", json.dumps(example, indent=2))
                print("Price prediction:", response.json(), "\n")
            else:
                print(f"Example {i}: Error {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Example {i}: Exception while calling the endpoint -> {e}")


if __name__ == "__main__":
    main()