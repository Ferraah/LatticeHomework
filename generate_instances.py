import random
import numpy as np
import argparse

def generate_instance(n: int, up: int):
    random.seed(43)
    # U
    #U:list = [random.randint(0,up)for _ in range(n)]
    # C
    C = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n, 1):
            C[i][j] = random.randint(0,1)
            C[j][i] = C[i][j]
            if i == j:
                C[i][j] = 0

    with open(f"pco_{n}.txt", "+w") as f:
        f.write("N\n")
        f.write(f"{n}\n")
        f.write("U\n")
        for i in range(n):
            f.write(f"{i};{up}\n")
        
        f.write("C\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{i},{j};{C[i][j]}\n")
    return 

def main():
    parser = argparse.ArgumentParser(description="Generate instances for Pccp.")
    parser.add_argument('num_instances', type=int, help='Number of instances to generate')
    parser.add_argument('upper_bound', type=int, help='Upper bound for random integers in U')
    args = parser.parse_args()

    generate_instance(args.num_instances, args.upper_bound)

if __name__ == '__main__':
    main()