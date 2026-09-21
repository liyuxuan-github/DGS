import numpy as np
import os


def vendi_score(data, sigma=None, alpha=1, kernel='rbf'):
    n = data.shape[0]

    if n <= 1:
        return float(n) if n == 1 else 0.0

    if kernel == 'rbf':
        dists_sq = np.sum(
            (data[:, None, :] - data[None, :, :]) ** 2,
            axis=2
        )

        if sigma is None:
            mask = ~np.eye(n, dtype=bool)
            median_dist = np.median(np.sqrt(dists_sq[mask]))
            sigma = median_dist / np.sqrt(2) if median_dist > 0 else 1.0

        K = np.exp(-dists_sq / (2 * sigma ** 2))

    elif kernel == 'cosine':
        data_norm = data / (
            np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
        )
        K = np.dot(data_norm, data_norm.T)
        K = np.clip(K, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    K_norm = K / n

    eigvals = np.linalg.eigvalsh(K_norm)
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 1.0

    if alpha == 1:
        H = -np.sum(eigvals * np.log(eigvals))
    else:
        H = (
            1 / (1 - alpha)
        ) * np.log(np.sum(eigvals ** alpha))

    return float(np.exp(H))


if __name__ == "__main__":
    print("=" * 60)
    print("Vendi Score Results")
    print("=" * 60)

    for file in sorted(os.listdir('./')):
        if not file.endswith('.npy'):
            continue

        data = np.load(file)
        n_samples = data.shape[0]

        vendi_rbf = vendi_score(
            data,
            sigma=None,
            alpha=1,
            kernel='rbf'
        )

        vendi_cos = vendi_score(
            data,
            sigma=None,
            alpha=1,
            kernel='cosine'
        )

        data_norm = data / (
            np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
        )
        cos_sim = np.dot(data_norm, data_norm.T)
        cos_dist = 1 - cos_sim
        mean_cos_dist = float(np.mean(cos_dist))

        print(f"\nFile: {file}  (Samples: {n_samples})")
        print(
            f"  Mean Cosine Distance:        "
            f"{mean_cos_dist:.4f}"
        )
        print(
            f"  Vendi Score (RBF Kernel):    "
            f"{vendi_rbf:.4f}  "
            f"[Range: 1 ~ {n_samples}]"
        )
        print(
            f"  Vendi Score (Cosine Kernel): "
            f"{vendi_cos:.4f}  "
            f"[Range: 1 ~ {n_samples}]"
        )