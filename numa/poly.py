def degrees(n: int):
    degrees_ = []
    for i in range(n):
        m = i + 1
        for j in reversed(range(m + 1)):
            degrees_.append((j, m - j))
    return degrees_
