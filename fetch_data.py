import requests
import json

def fetch_all_data():
    # Base URL for the Flask application
    base_url = "http://localhost:5001"

    # List of queries to search for
    queries = [
        "constitution",
        "fundamental rights",
        "criminal law",
        "civil procedure",
        "family law",
        "property law",
        "contract law",
        "tax law",
        "company law",
        "environmental law"
    ]

    # Fetch data for each query
    for query in queries:
        print(f"\nFetching data for query: {query}")

        # Fetch acts data
        try:
            response = requests.post(
                f"{base_url}/fetch-acts",
                json={"query": query}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result['message']}")
                print(f"Total acts saved: {len(result['acts'])}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error fetching acts data: {str(e)}")

if __name__ == "__main__":
    print("Starting data fetch process...")
    fetch_all_data()
    print("\nData fetch process completed!")
