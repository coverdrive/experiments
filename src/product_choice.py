import numpy as np
from random import sample
from typing import List


def get_softmax_probs(scores: np.ndarray) -> np.ndarray:
    aug_scores = np.insert(scores, 0, 0.)
    adj_scores = aug_scores - np.max(aug_scores)
    return np.exp(adj_scores) / sum(np.exp(adj_scores))


def get_round_scores(
    utils: np.ndarray,
    round_product_feature_vals: np.ndarray,
    round_prices: np.ndarray
) -> np.ndarray:
    return np.matmul(round_product_feature_vals, utils) - round_prices


def get_rounds_choices(
    rp: List[List[int]],
    fu: np.ndarray,
    pfv: np.ndarray,
    p: np.ndarray,
) -> List[int]:
    return [np.random.choice(
        [-1] + r,
        1,
        p=get_softmax_probs(
            get_round_scores(
                fu,
                pfv[r],
                p[r]
            )
        )
    )[0] for r in rp]


def get_rounds_products(
    rounds: int,
    products: int,
    round_products: int
) -> List[List[int]]:
    return [sample(range(products), round_products) for _ in range(rounds)]


def get_kronecker_delta(
    product_indices: List[int],
    chosen_product: int
) -> np.ndarray:
    return np.array([1. if chosen_product == i else 0.
                       for i in product_indices])


def get_gradient_log_prob(
    theta: np.ndarray,
    rp: List[List[int]],
    rc: List[int],
    pfv: np.ndarray,
    p: np.ndarray
) -> np.ndarray:
    return sum(np.matmul(
        pfv[r].T,
        get_kronecker_delta(r, rc[i]) -
        get_softmax_probs(
            get_round_scores(
                theta,
                pfv[r],
                p[r]
            )
        )[1:]
    ) for i, r in enumerate(rp)) / len(rp)


if __name__ == '__main__':
    num_rounds = 100000
    num_products = 20
    num_features = 4
    num_round_products = 12
    feature_utils = 10. * np.random.rand(num_features)
    product_feature_vals = 100. * np.random.rand(num_products, num_features)
    print(product_feature_vals)
    base_prices = np.matmul(product_feature_vals, feature_utils)
    prices = np.array([np.random.normal(bp, 2.) for bp in base_prices])
    print(prices)
    rounds_products = get_rounds_products(
        num_rounds,
        num_products,
        num_round_products
    )
    print(rounds_products)
    rounds_choices = get_rounds_choices(
        rounds_products,
        feature_utils,
        product_feature_vals,
        prices
    )
    print(rounds_choices)
    print(feature_utils)

    init_theta = np.empty(num_features)
    init_theta.fill(5.0)
    print(init_theta)
    theta = init_theta
    step_size = 0.003
    for _ in range(100):
        grad = get_gradient_log_prob(
            theta,
            rounds_products,
            rounds_choices,
            product_feature_vals,
            prices
        )
        print(grad)
        theta += step_size * grad
        print(theta)
        print("------")

