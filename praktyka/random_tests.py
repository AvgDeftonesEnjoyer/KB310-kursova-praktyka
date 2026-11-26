#!/usr/bin/env python3
"""Навчальна програма: чотири статистичні тести бітових послідовностей.

Запуск: python3 random_tests.py         # 6 прикладів
        python3 random_tests.py --all   # 402 послідовності
Без сторонніх бібліотек. Це не повна реалізація NIST STS.
"""

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

ALPHA = 0.01
BLOCK_SIZE = 1000
SERIAL_M = 2
SEED = 20260925


def gammaincc(a, x):
    """Регуляризована верхня неповна гамма-функція Q(a,x).

    Використовується для p-value в тестах з розподілом хі-квадрат.
    Чисельні формули: розклад у ряд або неперервний дріб.
    """
    if a <= 0 or x < 0:
        raise ValueError("Потрібні a > 0 та x >= 0")
    if x == 0:
        return 1.0
    lg = math.lgamma(a)
    eps = 1e-13
    tiny = 1e-300
    if x < a + 1:
        # P(a,x) за збіжним рядом, потім Q=1-P.
        acc = 1.0 / a
        summand = acc
        for k in range(1, 10000):
            summand *= x / (a + k)
            acc += summand
            if abs(summand) < abs(acc) * eps:
                break
        p = acc * math.exp(-x + a * math.log(x) - lg)
        return max(0.0, min(1.0, 1.0 - p))
    # Q(a,x) за модифікованим неперервним дробом (Lentz).
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny) * (1 if b >= 0 else -1)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = math.exp(-x + a * math.log(x) - lg) * h
    return max(0.0, min(1.0, q))


def frequency_test(bits):
    """Monobit: чи близька кількість одиниць до n/2?"""
    n = len(bits)
    if n == 0:
        raise ValueError("Порожня послідовність")
    s = 2 * bits.count("1") - n
    return math.erfc(abs(s) / math.sqrt(2.0 * n))


def block_frequency_test(bits, block_size=BLOCK_SIZE):
    """Частотний тест усередині неперетинних блоків."""
    k = len(bits) // block_size
    if k < 1:
        raise ValueError("Послідовність коротша за один блок")
    chi_sq = 0.0
    for j in range(k):
        piece = bits[j * block_size: (j + 1) * block_size]
        p_one = piece.count("1") / block_size
        chi_sq += (p_one - 0.5) ** 2
    chi_sq *= 4.0 * block_size
    return gammaincc(k / 2.0, chi_sq / 2.0)


def runs_test(bits):
    """Runs: чи типова кількість груп однакових сусідніх бітів?"""
    n = len(bits)
    pi = bits.count("1") / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0  # Попередня вимога балансу не виконана.
    v = 1 + sum(bits[i] != bits[i - 1] for i in range(1, n))
    denom = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    return math.erfc(abs(v - 2.0 * n * pi * (1 - pi)) / denom)


def psi2(bits, m):
    """Статистика частот циклічних m-бітових шаблонів."""
    if m == 0:
        return 0.0
    n = len(bits)
    counts = [0] * (1 << m)
    mask = (1 << m) - 1
    window = 0
    for i, ch in enumerate(bits + bits[:m - 1]):
        window = ((window << 1) | (ch == "1")) & mask
        if i >= m - 1:
            counts[window] += 1
    return ((1 << m) / n) * sum(count * count for count in counts) - n


def serial_test(bits, m=SERIAL_M):
    """Serial: повертає ДВА p-value; обидва мають пройти поріг."""
    if m < 2:
        raise ValueError("m має бути не меншим за 2")
    x_m = psi2(bits, m)
    x_m1 = psi2(bits, m - 1)
    x_m2 = psi2(bits, m - 2)
    delta1 = max(0.0, x_m - x_m1)
    delta2 = max(0.0, x_m - 2.0 * x_m1 + x_m2)
    return (gammaincc(2 ** (m - 2), delta1 / 2.0),
            gammaincc(2 ** (m - 3), delta2 / 2.0))


def generate_bits(kind, n, seed):
    """Створює зразок. random.Random НЕ є криптостійким генератором."""
    rng = random.Random(seed)
    if kind == "prng":
        return format(rng.getrandbits(n), f"0{n}b")
    if kind == "global_bias":
        return "".join("1" if rng.random() < 0.54 else "0" for _ in range(n))
    if kind == "local_bias":
        half = n // 2
        return ("".join("1" if rng.random() < 0.54 else "0" for _ in range(half))
                + "".join("1" if rng.random() < 0.46 else "0" for _ in range(n-half)))
    if kind == "alternating":
        return ("01" * (n // 2 + 1))[:n]
    if kind == "periodic_0011":
        return ("0011" * (n // 4 + 1))[:n]
    if kind == "zeros":
        return "0" * n
    raise ValueError(f"Невідомий тип: {kind}")


def cases(full):
    if full:
        # 100 + 100 + 100 + 100 + 1 + 1 = 402 зразки.
        groups = [("prng", 100), ("global_bias", 100),
                  ("local_bias", 100), ("alternating", 100),
                  ("periodic_0011", 1), ("zeros", 1)]
    else:
        groups = [(x, 1) for x in ("prng", "global_bias", "local_bias",
                                    "alternating", "periodic_0011", "zeros")]
    for group_index, (kind, count) in enumerate(groups):
        for i in range(count):
            yield kind, i + 1, SEED + group_index * 10000 + i


def check_bits(bits):
    p_frequency = frequency_test(bits)
    p_block = block_frequency_test(bits)
    p_runs = runs_test(bits)
    p_serial1, p_serial2 = serial_test(bits)
    flags = (p_frequency >= ALPHA, p_block >= ALPHA,
             p_runs >= ALPHA, min(p_serial1, p_serial2) >= ALPHA)
    return {
        "p_frequency": p_frequency,
        "p_block_frequency": p_block,
        "p_runs": p_runs,
        "p_serial_1": p_serial1,
        "p_serial_2": p_serial2,
        "pass_frequency": int(flags[0]),
        "pass_block_frequency": int(flags[1]),
        "pass_runs": int(flags[2]),
        "pass_serial": int(flags[3]),
        "pass_all": int(all(flags)),
    }


def main():
    parser = argparse.ArgumentParser(description="4 статистичні тести випадкових послідовностей")
    parser.add_argument("--all", action="store_true", help="Всі 402 послідовності замість 6 демонстрацій")
    parser.add_argument("--n", type=int, default=100000, help="Довжина однієї послідовності в бітах")
    parser.add_argument("--out", default=None, help="Шлях до CSV-таблиці з результатами")
    args = parser.parse_args()
    if args.n < BLOCK_SIZE or args.n < 8:
        parser.error(f"--n має бути >= {BLOCK_SIZE}")
    result_path = Path(args.out or ("results_all.csv" if args.all else "results_demo.csv"))
    summary_path = result_path.with_name(result_path.stem + "_summary.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (kind, number, seed) in enumerate(cases(args.all), start=1):
        bits = generate_bits(kind, args.n, seed)
        stat = check_bits(bits)
        row = {"sample": index, "type": kind, "number": number, "n_bits": args.n,
               "ones": bits.count("1"), "zeros": bits.count("0"), **stat}
        rows.append(row)
        if not args.all:
            short = ", ".join(f"{t}={'OK' if row['pass_'+t] else 'FAIL'}"
                              for t in ("frequency", "block_frequency", "runs", "serial"))
            print(f"{kind:16} | {short}")
        elif index % 50 == 0:
            print(f"Опрацьовано {index} / 402 послідовностей...")

    with result_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    statnames = ("frequency", "block_frequency", "runs", "serial", "all")
    sums = defaultdict(lambda: {"count": 0, **{x: 0 for x in statnames}})
    for row in rows:
        g = sums[row["type"]]
        g["count"] += 1
        for name in statnames:
            g[name] += row["pass_" + name]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "samples"] + ["pass_" + x for x in statnames])
        for kind, stats in sums.items():
            writer.writerow([kind, stats["count"]] + [stats[x] for x in statnames])
    print(f"\nРезультати: {result_path.resolve()}")
    print(f"Зведення:   {summary_path.resolve()}")
    print("Увага: прийняття тесту не доводить криптографічну безпеку.")


if __name__ == "__main__":
    main()
