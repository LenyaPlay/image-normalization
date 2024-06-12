from os import walk
import cv2
from cv2.typing import MatLike, Size
from pathlib import Path
import numpy as np


def feature_projection_yamashita(img: MatLike,
                                 threshold: float = 128) -> MatLike:
    black_white = np.where(img <= threshold, 1, 0)
    h = black_white.sum(axis=1)
    v = black_white.sum(axis=0)
    return h, v


def normalize(img: MatLike,
              new_size: Size | None = None,
              alpha_h: int = 0,
              alpha_v: int = 0):
    J, I = M, N = img.shape
    if new_size is not None:
        N, M = new_size

    h, v = feature_projection_yamashita(img)
    h += alpha_h
    v += alpha_v

    h_cum = np.cumsum(h)
    v_cum = np.cumsum(v)

    m_nod = (M - 1) / h_cum[-1]
    n_mod = (N - 1) / v_cum[-1]

    m = h_cum * m_nod
    n = v_cum * n_mod

    m = m.astype(np.int64)
    n = n.astype(np.int64)

    out = np.full((M + 1, N + 1), img.max())

    # for j in range(J):
    #     for i in range(I):
    #         out[m[j], n[i]] = img[j, i]

    for j in range(1, J):
        for i in range(1, I):
            m_from = m[j - 1]
            m_to = m[j]
            n_from = n[i - 1]
            n_to = n[i]
            out[m_from: m_to + 1, n_from: n_to + 1] = img[j, i]
    return out


def process():
    input_dir = "./data/in"
    output_dir = "./data/out"
    for (dirpath, dirnames, filenames) in walk(input_dir):
        for filename in filenames:
            img = cv2.imread(f'{dirpath}/{filename}', cv2.IMREAD_GRAYSCALE)

            clean_dir = dirpath.replace(input_dir, "", 1)
            Path(f'{output_dir}/{clean_dir}').mkdir(parents=True, exist_ok=True)

            normalized = normalize(img)
            cv2.imwrite(f'{output_dir}/{clean_dir}/{filename}', normalized)


if __name__ == "__main__":
    process()
