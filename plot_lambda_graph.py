import matplotlib.pyplot as plt

gpt_4o_corr = 0.46
gpt_4o_score = 9.5340
distill_corr = 0.44
distill_score = 8.8920
dpo_corrs = [0.48, 0.53, 0.63, 0.74, 0.73]
dpo_scores = [9.2840, 9.5180, 9.3840, 9.2660, 8.8920]
lambdas = [0, 0.25, 0.5, 0.75, 1]

def main():
    plt.figure(figsize=(14, 5))
    plt.rcParams.update({'font.size': 24})
    linewidth = 5
    markersize = 10

    # Plotting dpo_corrs over lambdas
    plt.subplot(1, 2, 1)
    plt.axhline(y=gpt_4o_corr, color='#DDCC77', linestyle='--', label='GPT-4o', linewidth=linewidth)
    plt.axhline(y=distill_corr, color='#332288', linestyle='--', label='Distill', linewidth=linewidth)
    plt.plot(lambdas, dpo_corrs, color="#44AA99", marker='o', label='DPO', linewidth=linewidth, markersize=markersize)
    plt.xlabel('λ')
    plt.ylabel('Student Outcomes')
    plt.grid(True)

    # Plotting dpo_scores over lambdas
    plt.subplot(1, 2, 2)
    plt.axhline(y=gpt_4o_score, color='#DDCC77', linestyle='--', label='GPT-4o', linewidth=linewidth)
    plt.axhline(y=distill_score, color='#332288', linestyle='--', label='Distill', linewidth=linewidth)
    plt.plot(lambdas, dpo_scores, color="#44AA99", marker='o', label='DPO', linewidth=linewidth, markersize=markersize)
    plt.xlabel('λ')
    plt.ylabel('Pedagogical Principles')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("../figures/lambda_exp.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
