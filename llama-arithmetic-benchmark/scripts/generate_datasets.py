import random
import csv
import os
import argparse
import sys

def generate_integer_arithmetic(num_samples=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    problems = []
    for _ in range(num_samples):
        op = random.choice(['+', '-'])
        a, b = random.randint(-100, 100), random.randint(-100, 100)
        ans = a + b if op == '+' else a - b
        problems.append({"problem": f"{a} {op} {b}", "answer": ans})
    return problems

def generate_decimal_arithmetic(num_samples=1000, seed=None):
    if seed is not None:
        random.seed(seed + 1)
    problems = []
    for _ in range(num_samples):
        op = random.choice(['+', '-'])
        a, b = round(random.uniform(-100, 100), 2), round(random.uniform(-100, 100), 2)
        ans = round(a + b, 2) if op == '+' else round(a - b, 2)
        problems.append({"problem": f"{a} {op} {b}", "answer": ans})
    return problems

def save_to_csv(problems, filename):
    os.makedirs('data', exist_ok=True)
    path = os.path.join('data', filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['problem', 'answer'])
        writer.writeheader()
        writer.writerows(problems)
    print(f"Saved {len(problems)} problems to {path}")

def main(num_samples=1000, seed=None):
    int_probs = generate_integer_arithmetic(num_samples, seed)
    dec_probs = generate_decimal_arithmetic(num_samples, seed)
    save_to_csv(int_probs, 'integer_arithmetic.csv')
    save_to_csv(dec_probs, 'decimal_arithmetic.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arithmetic datasets.")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples per dataset (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(args=[])

    main(num_samples=args.n, seed=args.seed)
