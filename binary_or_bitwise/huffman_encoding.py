"""Huffman encoding implementation.

Just for study purposes. It is not fully optimized and will
not save the Encoding Table alongside the compressed file.
Therefore, decompression will be only possible using the
same ``Huffman`` instance used to compress the file (or
another ``Huffman`` instance loaded with the same Encoding
table.)
"""
import typing as t
import heapq
import re

import numpy as np


class Huffman:
    """Encoding and decoding of files using Huffman Encoding."""

    def __init__(self):
        self.huff_enc_table = None  # type: t.Dict[str, t.Tuple[int, int]]
        self.orig_file_size = -1
        self._num_symbols = -1
        self._total_nodes = -1
        self._tree_root_ind = -1
        self._max_len_code = -1
        self._huff_tree_son_l = None  # type: np.ndarray
        self._huff_tree_son_r = None  # type: np.ndarray
        self._huff_tree_symbol = None  # type: np.ndarray
        self._re_special_char = re.compile("(\n|\r)")

    def __str__(self) -> str:
        divisor = "+---+-{}-+".format(self._max_len_code * "-")
        res = [divisor]

        for symb in self.huff_enc_table:
            code = self._encode_symb(symb)
            enc_symb = self._re_special_char.sub("?", symb)
            res.append(r"| {} | {:<{fill}} |".format(
                enc_symb, code, fill=self._max_len_code))

        res.append(divisor)

        return "\n".join(res)

    def _huff_tree_walk(self, cur_node_ind: int, prefix_code: int, mask: int,
                        depth: int) -> None:
        """Build a symbol table walking recursively in the Huffman tree."""
        if cur_node_ind < self._num_symbols:
            self.huff_enc_table[self._huff_tree_symbol[cur_node_ind]] = (
                prefix_code, depth)
            self._max_len_code = max(self._max_len_code, depth)

        else:
            self._huff_tree_walk(
                cur_node_ind=self._huff_tree_son_l[cur_node_ind],
                prefix_code=prefix_code,
                mask=mask << 1,
                depth=depth + 1)

            self._huff_tree_walk(
                cur_node_ind=self._huff_tree_son_r[cur_node_ind],
                prefix_code=prefix_code | mask,
                mask=mask << 1,
                depth=depth + 1)

    def _build_huff_table(self,
                          symb_freq: t.Tuple[np.array, np.ndarray]) -> None:
        """Build Huffman Encoding Table."""
        self._num_symbols = symb_freq[0].size

        self._total_nodes = 2 * self._num_symbols - 1
        self._tree_root_ind = self._total_nodes - 1

        self._huff_tree_son_l = np.full(self._total_nodes, -1)
        self._huff_tree_son_r = np.full(self._total_nodes, -1)
        self._huff_tree_symbol = np.full(self._num_symbols, "")

        p_queue = []  # type: t.List[t.Tuple[int, int]]

        for ind, item in enumerate(zip(*symb_freq)):
            symb, freq = item
            heapq.heappush(p_queue, (freq, ind))
            self._huff_tree_symbol[ind] = symb

        cur_node_id = self._num_symbols
        while cur_node_id < self._total_nodes:
            fr_a, ind_a = heapq.heappop(p_queue)
            fr_b, ind_b = heapq.heappop(p_queue)

            new_item = (fr_a + fr_b, cur_node_id)

            heapq.heappush(p_queue, new_item)
            self._huff_tree_son_l[cur_node_id] = ind_a
            self._huff_tree_son_r[cur_node_id] = ind_b

            cur_node_id += 1

        self.huff_enc_table = {}
        self._huff_tree_walk(
            cur_node_ind=self._tree_root_ind, prefix_code=0, mask=1, depth=0)

    def _encode_symb(self, symb: str) -> str:
        """Encode a given ``symbol`` to its variable-length code."""
        code_num, code_len = self.huff_enc_table[symb]
        code_bin = bin(code_num)[2:]

        if len(code_bin) < code_len:
            code_bin += "0" * (code_len - len(code_bin))

        return code_bin

    def encode(self, path_input: str,
               path_output: t.Optional[str] = None) -> float:
        """Compress file using Huffman Encoding.

        Arguments
        ---------
        path_input : :obj:`str`
            Input file path.

        path_out : :obj:`str`, optional
            Output file path. If not given, will print the result on
            stdandard output (stdout).

        Returns
        -------
        :obj:`float`
            Compression fator (1 - compressed_size / original_size).
        """
        compressed_file_tokens = []  # t.List[int]

        batch_pos = 0
        compressed_byte = 0

        with open(path_input) as f_in:
            f_content = f_in.read()
            self.orig_file_size = len(f_content)

        symb_freq = np.unique(list(f_content), return_counts=True)
        self._build_huff_table(symb_freq)

        for symb in f_content:
            code_num, code_len = self.huff_enc_table[symb]
            code_mask = 1

            for _ in np.arange(code_len):
                new_bit = 1 if code_num & code_mask else 0
                compressed_byte |= new_bit << batch_pos
                code_mask <<= 1
                batch_pos += 1

                if batch_pos >= 8:
                    compressed_file_tokens.append(compressed_byte)
                    batch_pos = 0
                    compressed_byte = 0

        if batch_pos != 0:
            compressed_file_tokens.append(compressed_byte)

        compressed_file = "".join(map(chr, compressed_file_tokens))

        if path_output is None:
            print(compressed_file)

        else:
            with open(path_output, "w") as f_out:
                f_out.write(compressed_file)

        return 1.0 - len(compressed_file) / len(f_content)

    def decode(self, path_input: str,
               path_output: t.Optional[str] = None) -> float:
        """Inflate file compressed with Huffman Encoding.

        Arguments
        ---------
        path_input : :obj:`str`
            Compressed input file path.

        path_out : :obj:`str`, optional
            Output file path. If not given, will print the result on
            stdandard output (stdout).

        Returns
        -------
        :obj:`float`
            Inflation fator (original_size / compressed_size).
        """
        decompressed_file_tokens = []  # type: t.List[str]

        with open(path_input) as f_in:
            f_content = f_in.read()

        cur_tree_ind = self._tree_root_ind
        for byte in f_content:
            mask = 1
            bin_byte = ord(byte)

            while mask != 256:
                if cur_tree_ind < self._num_symbols:
                    decompressed_file_tokens.append(
                        self._huff_tree_symbol[cur_tree_ind])
                    cur_tree_ind = self._tree_root_ind

                if bin_byte & mask:
                    cur_tree_ind = self._huff_tree_son_r[cur_tree_ind]

                else:
                    cur_tree_ind = self._huff_tree_son_l[cur_tree_ind]

                mask <<= 1

        original_file = "".join(decompressed_file_tokens[:self.orig_file_size])

        if path_output is None:
            print(original_file)

        else:
            with open(path_output, "w") as f_out:
                f_out.write(original_file)

        return len(original_file) / len(f_content)


def _test() -> None:
    import sys

    if len(sys.argv) < 2:
        print("usage: python", sys.argv[0], "<input path> [output path]")
        exit(1)

    path_inp = sys.argv[1]

    try:
        path_out = sys.argv[2]

    except IndexError:
        path_out = sys.argv[1]

    huf = Huffman()
    print("Compressed: {:.4f}".format(
        huf.encode(path_input=path_inp, path_output=path_out + "_compressed")))
    print(huf)
    print("Inflated: {:.4f}".format(
        huf.decode(
            path_input=path_inp + "_compressed",
            path_output=path_out + "_decompressed")))


if __name__ == "__main__":
    _test()
