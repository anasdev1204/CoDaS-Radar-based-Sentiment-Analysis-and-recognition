    input:
    - modality [IMU, InfraCam, Radar]
    - sentiment labels [(focus, distraction), stress, relaxation]
    - duration [4, 6, 8, 12]
    - overlap {0, 25, 50, 75}
    - method of representation [RAW, 1D, 2D]

    output:
    - window-level samples per modality and per representation

---

    for each participant P in campaign 3:
        for each sentiment episode E (one phase) of participant P:
            0. obtain episode-level label L(E)
            1. load raw signal for modality M during episode E -> S(E)
            2. modality-specific cleaning + temporal regularization
                for IMU or infra cam:
                    sensor fusion, resample to 100 Hz
                for radar:
                    normalize to fix points number
            3. split the signal S(E) into overlapping windows:
                window length = W seconds
                duration = S seconds
                for each window w in episode E:
                    4. assign window label L(w) = L(E)
                    5. normalization (within-window)
                    6. representation extraction
                        if R == RAW:
                            multivariate time-series
                        if R == 1D:
                            feature vector (mean/std/energy/etc.)
                        if R == 2D:
                            image-like representation (e.g., similarity/recurrence matrix, per-channel then stacked as channels)
