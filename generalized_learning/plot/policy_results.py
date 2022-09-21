

import pathlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_policy_results(policy_results_file):

    policy_results_file = pathlib.Path(policy_results_file)

    policy_results = pd.read_csv(policy_results_file.as_posix())
    policy_results["solved"] = policy_results["solved"].astype(int)

    df = policy_results.groupby("abstraction")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for column, ylabel, ax in [("solved", "Success rate", axes[0]),
                               ("cost", "Avg. cost", axes[1])]:

        mean = df[column].mean()
        stdev = df[column].std()

        mean.plot.bar(yerr=stdev, ax=ax, rot=45)
        ax.set_xlabel("Policy type")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig("%s/policy_results.png" % (policy_results_file.parent))


if __name__ == "__main__":

    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-file", required=True)

    args = argument_parser.parse_args()

    plot_policy_results(args.input_file)
