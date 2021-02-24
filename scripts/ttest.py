import json
import math
import numpy
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from scipy.stats import norm
from tabulate import tabulate

def _test(a_data, b_data):
    return ttest_ind(a_data, b_data)

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def test(a_filename, b_filename):
    return _test(a_data, b_data)

def num_samples_AB(mean_A, mean_B, var_A, var_B, alpha=0.05, beta=0.2, max_samples=2**22):
    """
    Assuming same # samples A and B, and normal distribution
    
    alpha is rate of type I error
    beta is (1 - test power)
    https://towardsdatascience.com/required-sample-size-for-a-b-testing-6f6608dd330a
    """
    Za = norm.ppf(1 - alpha / 2)
    Zb = norm.ppf(1 - beta) 
   
    if mean_A == mean_B:
        return max_samples
    
    n = (var_A + var_B) * (Za + Zb) ** 2
    n /= (mean_A - mean_B) ** 2
    n = int(math.ceil(n))

    return n

def power_AB(mean_A, mean_B, var_A, var_B, n_A, n_B, alpha=0.05):
    Za = norm.ppf(1 - alpha / 2)
    delta = numpy.abs(mean_A - mean_B) 
    power = delta / np.sqrt(var_A / n_A + var_B / n_B )
    power -= Za
    return power 

def generate_sample(mean, var):
    return numpy.random.normal(mean, var)


def plot_hist(title, a_data, b_data):
    mu = 0.1
    sigma = 0.1
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle(title)
    for data, color in zip([a_data, b_data], ['r', 'b']):
        count, bins, ignored = plt.hist(data, 30, density=True)
        # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                    # np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                # linewidth=2, color=color)
    plt.show()

def compare_cases(p_thresh=0.001):
    cases = {
        "same": {
            "A": (0.1, 0.1),
            "B": (0.1, 0.1),
        },
        "same-mean-diff-var": {
            "A": (0.1, 0.1),
            "B": (0.1, 0.2),
        },
        "diff-mean-tight-var": {
            "A": (0.1, 0.1),
            "B": (0.2, 0.1),
        },
        "diff-mean-loose-var": {
            "A": (0.1, 0.2),
            "B": (0.2, 0.2),
        },
        "5pctdiff-mean-loose-var": {
            "A": (0.1, 0.2),
            "B": (0.105, 0.2),
        },
        "1pctdiff-mean-loose-var": {
            "A": (0.1, 0.2),
            "B": (0.101, 0.2),
        },

    }
    
    n_trials = 100
    failed = 0
    for trial in range(n_trials):
        try:
            for case in cases:
                a_mean, a_var = cases[case]["A"]
                b_mean, b_var = cases[case]["B"]
                n_samples = num_samples_AB(a_mean, b_mean, a_var, b_var)
                a_data = numpy.random.normal(a_mean, a_var, n_samples)
                b_data = numpy.random.normal(b_mean, b_var, n_samples)
                # print(f"{case}: n_samples={n_samples}")
                res = ttest_ind(a_data, b_data)
                assert not math.isnan(res.pvalue)
                
                diff = res.pvalue < p_thresh
                # plot_hist(case, a_data, b_data)
                assert (a_mean == b_mean) == (not diff), f"False conclusion for {case}, {res}"

        except Exception as e:
            failed += 1
            print(e)

        
    print(f"passed {n_trials - failed} out of {n_trials} times")

def compare_repeatruns():
    ROOT = Path("/Users/whc/Downloads/v1_repeatruns_nopstate2500")
    A = ROOT / "20210111_184944_.json"
    B = ROOT / "20210112_021614_.json"

    a_json = load_json(A)
    b_json = load_json(B)

    info = []
    total_runtime_s = 0
    for a_bench, b_bench in zip(a_json["benchmarks"], b_json["benchmarks"]): 
        name = a_bench["name"]
        assert name == b_bench["name"] 
        a_data = a_bench["stats"]["data"]
        b_data = b_bench["stats"]["data"]
        a_mean, a_var = np.mean(a_data), np.var(a_data)
        b_mean, b_var = np.mean(b_data), np.var(b_data)
        n_samples = num_samples_AB(a_mean, a_mean + a_mean * 0.01, a_var, b_var)
        runtime_s = n_samples * (a_mean + b_mean)
        total_runtime_s += runtime_s
        info.append({
            "name": name,
            "mean": a_mean,
            "var": a_var,
            "len(data)": len(a_data),
            "samples_needed": n_samples,
            "runtime_s": runtime_s,
            "power": power_AB(a_mean, a_mean + a_mean * 0.01, a_var, b_var, len(a_data), len(b_data))
        })
    info.sort(key=lambda x: x["runtime_s"])
    print(tabulate(info, headers="keys"))
    print(f"total runtime would be {total_runtime_s / 3600.} hours")


if __name__ == "__main__":
    compare_repeatruns()
    # compare_cases()


"""
1. implement and verify #samples calculator, show it works 9/10 times on generated normal data
2. try using it on benchmark, see how often it works on A/A
  use existing json data first? 

    ** add a mode that computes the confidence interval given the actual #samples
  ultimately run live collections?
    set up 2 conda envs and build the process infra

3. then try A/B with some different performance changes

"""