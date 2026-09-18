from predict import get_prediction

def run_cli():
    print("--- Premier League Match Predictor ---")
    while True:
        home = input("\nEnter Home Team (or 'exit'): ").strip()
        if home.lower() == 'exit':
            break
        away = input("Enter Away Team: ").strip()

        try:
            res = get_prediction(home, away)
            print(f"\nPredicted Result: {res['prediction']}")
            print("Probabilities:")
            for outcome, prob in res['probabilities'].items():
                print(f"  {outcome}: {prob}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    run_cli()