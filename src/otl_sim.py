from scipy.stats import poisson
from typing import Sequence, Mapping, Any
import numpy as np


def get_casepacks_from_policy_and_eip(
    pol: Sequence[int],
    eip: int
) -> int:
    try:
        cp = next(i for i, x in enumerate(pol) if eip >= x)
    except StopIteration:
        cp = len(pol)
    return cp


def simulate_policy(
    demand: float,
    lead_time: int,
    pog: int,
    casepack: int,
    policy: Sequence[int],
    sim_days: int
) -> Sequence[Mapping[str, Any]]:
    """
    demand is the mean of a poisson distribution for daily demand
    policy is descending-sorted sequence [i_0, i_1, ..., i_n].
    we order k casepacks if k is the smallest index such that
    Inventory Position >= i_k
    """

    all_demands = poisson(demand).rvs(sim_days)
    sim = []
    data = {"BOH": pog, "OW": [0] * lead_time, "BIP": pog}
    for d in all_demands:
        data["Demand"] = d
        data["Unmet Demand"] = max(0, d - data["BOH"])
        data["EOH"] = max(0, data["BOH"] - d)
        data["EIP"] = data["EOH"] + sum(data["OW"])
        order = get_casepacks_from_policy_and_eip(
            policy,
            data["EIP"]
        ) * casepack
        data["Order"] = order
        sim.append(data)
        # print(data)
        boh = data["EOH"] + data["OW"][0] if lead_time > 0 else order
        ow = data["OW"][1:] + [order] if lead_time > 0 else []
        data = {
            "BOH": boh,
            "OW": ow,
            "BIP": boh + sum(ow)
        }
    return sim


if __name__ == '__main__':
    this_demand = 0.12
    this_lead_time = 4
    this_pog = 4
    this_casepack = 8

    this_alpha = 0.5
    this_beta = 0.95

    poisson_mean = this_demand * (this_lead_time + 1)
    poisson_dist = poisson(poisson_mean)
    otl = np.ceil(this_alpha * this_pog + poisson_dist.ppf(this_beta))
    more_knots = int(np.floor(otl / this_casepack))
    this_policy = [otl - x * this_casepack for x in range(more_knots + 1)]

    num_sim_days = 10000
    sim_res = simulate_policy(
        demand=this_demand,
        lead_time=this_lead_time,
        pog=this_pog,
        casepack=this_casepack,
        policy=this_policy,
        sim_days=num_sim_days
    )
    # print(sim_res)
    holes_eoh = [max(0., 1. - s["EOH"] / this_pog) for s in sim_res]
    overflow_boh = [max(0., s["BOH"] / this_pog - 1.) for s in sim_res]
    demands = [s["Demand"] for s in sim_res]
    unmet_demands = [s["Unmet Demand"] for s in sim_res]

    print("OTL = %d" % otl)

    mean_holes_eoh = np.mean(holes_eoh)
    stdev_holes_eoh = np.std(holes_eoh)
    print("Mean Holes EOH = %.1f%%, Stdev Holes EOH = %.1f%%" %
          (mean_holes_eoh * 100, stdev_holes_eoh * 100))

    mean_overflow_boh = np.mean(overflow_boh)
    stdev_overflow_boh = np.std(overflow_boh)
    print("Mean Overflow BOH = %.1f%%, Stdev Overflow BOH = %.1f%%" %
          (mean_overflow_boh * 100, stdev_overflow_boh * 100))

    stockouts = sum(unmet_demands) / sum(demands)
    print("Stockouts = %.1f%%" % (stockouts * 100))
