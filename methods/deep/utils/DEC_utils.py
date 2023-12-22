

def target_distibution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T