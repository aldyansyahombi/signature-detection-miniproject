def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    """Character Error Rate = (S + D + I) / N  (jarak Levenshtein / panjang referensi)."""
    return levenshtein(pred, truth) / max(len(truth), 1)
