import typing as t

import numpy as np


class _TrieNode:
    def __init__(self, terminal: bool = False):
        self.keys = dict()  # type: t.Dict[str, _TrieNode]
        self.terminal = terminal

    def __getitem__(self, key: str) -> "_TrieNode":
        return self.keys[key]

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def insert(self, key: str) -> "_TrieNode":
        node = self.keys.get(key)

        if node is not None:
            return node

        node = _TrieNode()
        self.keys[key] = node

        return node

    def remove(self, key: str) -> "_TrieNode":
        return self.keys.pop(key)


class Trie:
    def __init__(self):
        self.head = _TrieNode()
        self._query = None

    def __contains__(self, query: str) -> bool:
        return self.search(query)

    def insert(self, query: str) -> None:
        cur_node = self.head

        for c in query:
            cur_node = cur_node.insert(c)

        cur_node.terminal = True

    def search(self, query: str, prefix: bool = False) -> bool:
        cur_node = self.head

        for c in query:
            if c not in cur_node:
                return False

            cur_node = cur_node[c]

        return cur_node.terminal or prefix

    def _remove(self, i: int, node: _TrieNode, remove_prefix: bool) -> bool:
        if i == len(self._query):
            node.terminal = False
            return len(node) == 0 or remove_prefix

        c = self._query[i]

        if c not in node:
            return False

        if self._remove(i + 1, node[c], remove_prefix):
            node.remove(c)

        return len(node) == 0

    def remove(self, query: str, remove_prefix: bool = False) -> None:
        if not query and remove_prefix:
            raise ValueError("Empty prefix can't be removed.")

        self._query = query
        self._remove(0, self.head, remove_prefix)
        self._query = None


def _test():
    num_tries = 800
    num_tests = 1000

    import random

    np.random.seed(16)
    random.seed(16)

    debug = False

    for tr in np.arange(num_tries):
        print(f"{tr + 1} / {num_tries} ...")
        trie = Trie()

        hist = set()

        for ts in np.arange(num_tests):
            print(f"  {ts + 1} / {num_tests} ...")
            size = np.random.randint(1, 99)
            inp = "".join(
                map(chr, np.random.randint(ord("a"), ord("z") + 1, size=size))
            )
            trie.insert(inp)

            assert inp in trie and trie.search(inp, prefix=False)

            hist.add(inp)

            if np.random.random() < 0.15:
                rand_item = hist.pop()
                remove_prefix = np.random.random() < 0.1667
                pref_start = (
                    np.random.randint(1 + len(rand_item))
                    if remove_prefix
                    else len(rand_item)
                )

                try:
                    trie.remove(rand_item[:pref_start], remove_prefix=remove_prefix)
                    assert rand_item not in trie and not trie.search(
                        rand_item, prefix=False
                    )

                except ValueError:
                    if not remove_prefix or pref_start > 0:
                        assert False

                    hist.add(rand_item)

                prefix = rand_item[:pref_start]

                if remove_prefix and prefix:
                    new_hist = set()

                    for s in hist:
                        if s.startswith(prefix):
                            assert s not in trie and not trie.search(
                                prefix, prefix=True
                            )

                        else:
                            new_hist.add(s)

                    hist = new_hist


if __name__ == "__main__":
    _test()
