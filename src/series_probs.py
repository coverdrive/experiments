from math import ceil


def factorial(n: int) -> int:
    return 1 if n == 0 else n * factorial(n-1)


def series_win_prob(
    n: int,
    w: float,
    l: float
) -> float:
    nf = factorial(n)
    return sum(nf / (factorial(i) * factorial(j) * factorial(n - i - j))
               * w ** i * l ** j * (1 - w - l) ** (n - i - j)
               for i in range(1, n + 1) for j in range(min(i, n - i + 1)))


def series_draw_prob(
    n: int,
    w: float,
    l: float
) -> float:
    nf = factorial(n)
    return sum(nf / (factorial(i) ** 2 * factorial(n - 2 * i))
               * (w * l) ** i * (1 - w - l) ** (n - 2 * i)
               for i in range(int(ceil((n + 1) / 2))))


if __name__ == '__main__':
    matches = 5
    win_prob = 0.31
    loss_prob = 0.47
    series_win = series_win_prob(
        n=matches,
        w=win_prob,
        l=loss_prob
    )
    series_loss = series_win_prob(
        n=matches,
        w=loss_prob,
        l=win_prob
    )
    series_draw = series_draw_prob(
        n=matches,
        w=win_prob,
        l=loss_prob
    )
    print("Series Win Probability = %.2f" % series_win)
    print("Series Loss Probability = %.2f" % series_loss)
    print("Series Draw Probability = %.2f" % series_draw)
