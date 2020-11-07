class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(u, v):
        pu = self.find(u)
        pv = self.find(v)
        self.parent[pu] = pv
