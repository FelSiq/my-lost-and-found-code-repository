"""Calculates and quiers a TF-IDF matrix from documents of a given directory."""
import typing as t
import os
import re

import numpy as np

RE_NON_LETTERS = re.compile(r"[^a-zA-Z\s]+")
RE_EMPTY_SPACE = re.compile(r"\s+")


def get_document_paths(source_dir: str, file_extension: t.Optional[str] = None
                       ) -> t.Sequence[str]:
    """Get all full filepaths of files within ``source_dir`` and its subdirectories."""
    if os.path.isfile(source_dir):
        raise ValueError("Given path is a file. Argument must be a directory.")

    file_list = []  # type: t.List[str]

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if file_extension is None or filename.endswith(file_extension):
                file_list.append(os.path.join(root, filename))

    return file_list


def filter_symbols(words: str, min_word_size: int = 3) -> t.Sequence[str]:
    """Tokezine given string ``words``, removing non-letter symbols and short words."""
    words = RE_NON_LETTERS.sub(' ', words)

    tokens = [
        token.lower() for token in RE_EMPTY_SPACE.split(words)
        if len(token) >= min_word_size
    ]

    return tokens


def read_input(source_dir: str, file_extension: t.Optional[str] = None
               ) -> t.Tuple[np.ndarray, t.Tuple[str, ...], np.ndarray]:
    """Build a (Document x Word) frequency matrix from the documents within ``source_dir``."""
    filepaths = get_document_paths(
        source_dir=source_dir, file_extension=file_extension)

    word_freqs = {}  # type: t.Dict[str, int]
    document_freqs = {
    }  # type: t.Dict[str, t.Tuple[t.Sequence[str], t.Sequence[int]]]
    all_words = set()  # type: t.Set[str]

    for filepath in filepaths:
        with open(filepath) as cur_file:
            file_content = filter_symbols(cur_file.read())
            words, freqs = np.unique(file_content, return_counts=True)

            for word, freq in zip(words, freqs):
                word_freqs[word] = word_freqs.get(word, 0) + freq

            document_freqs[filepath] = (words, freqs)
            all_words.update(words)

    all_words_fixed = tuple(all_words)

    del all_words

    freq_mat = np.zeros((len(filepaths), len(all_words_fixed)))

    for file_id, filepath in enumerate(document_freqs):
        for word, freq in zip(*document_freqs[filepath]):
            freq_mat[file_id, all_words_fixed.index(word)] += freq

    return np.array(filepaths), all_words_fixed, freq_mat


def calc_tf_idf_mat(freq_mat: np.ndarray,
                    calc_vector_norm: bool = True,
                    normalize_tf: bool = True,
                    add_one_idf: bool = False
                    ) -> t.Union[t.Tuple[np.ndarray, np.ndarray], t.
                                 Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Build the TD-IDF matrix from the (Document x Word) frequency matrix.

    Each entry of the TD-IDF matrix is calculated as follows:
    $$
        \text{tf-idf}_{i,j} = \text{freq_mat}_{i, j} \text{log}_{2}\frac{|D|}{d_{j}}
    $$
    Where |D| is the total number of documents used to build the ``freq_mat`' and
    $d_{j}$ is the number of documents that have at least one entry of the $j$th word.
    """
    num_docs, _ = freq_mat.shape
    idf_vec = add_one_idf + np.log2(num_docs / (freq_mat > 0).sum(axis=0))
    tf_idf_mat = freq_mat * idf_vec

    if normalize_tf:
        tf_idf_mat /= freq_mat.sum(axis=1, keepdims=True)

    if calc_vector_norm:
        vector_norm = np.linalg.norm(tf_idf_mat, ord=2, axis=1)
        return tf_idf_mat, idf_vec, vector_norm

    return tf_idf_mat, idf_vec


def cosine_similarity(vec_a: np.ndarray,
                      vec_b: np.ndarray,
                      vec_a_norm: t.Optional[float] = None,
                      vec_b_norm: t.Optional[float] = None) -> float:
    r"""Calculates the cosine similarity between ``vec_a`` and ``vec_b``.

    The consine similarity is calculated as follows:
    $$
        \text{cos_sim}(A, B) = \frac{AB}{||A||_{2} ||B||_{2}}
    $$
    """
    if vec_a_norm is None:
        vec_a_norm = np.linalg.norm(vec_a, ord=2)

    if vec_b_norm is None:
        vec_b_norm = np.linalg.norm(vec_b, ord=2)

    return np.dot(vec_a, vec_b) / (vec_a_norm * vec_b_norm)


def query(query_val: str,
          tf_idf_mat: np.ndarray,
          idf_vec: np.ndarray,
          files: np.ndarray,
          words: t.Tuple[str, ...],
          top_k: int = 1,
          doc_vec_lens: t.Optional[np.ndarray] = None) -> str:
    """."""
    if top_k <= 0:
        raise ValueError("'top_k' must be a positive value!")

    if not isinstance(top_k, int):
        raise TypeError("'top_k' must be a integer value!")

    tokens = filter_symbols(query_val)

    top_k = min(top_k, files.size)

    if top_k == files.size:
        return files.copy()

    vector_query = np.zeros(tf_idf_mat.shape[1])

    for token, token_freq in zip(*np.unique(tokens, return_counts=True)):
        try:
            word_index = words.index(token)
            vector_query[word_index] = token_freq * idf_vec[word_index]

        except ValueError:
            pass

    vector_query /= len(tokens)

    _vector_query_norm = np.linalg.norm(vector_query, ord=2)

    if doc_vec_lens is None:
        doc_vec_lens = np.linalg.norm(tf_idf_mat, ord=2, axis=1)

    similarities = np.array([
        cosine_similarity(
            vector_doc,
            vector_query,
            vec_a_norm=doc_vec_len,
            vec_b_norm=_vector_query_norm)
        for doc_vec_len, vector_doc in zip(doc_vec_lens, tf_idf_mat)
    ])

    if top_k == 1:
        return files[similarities.argmax()]

    return files[np.argsort(similarities) < top_k]


def _test():
    import sys
    if len(sys.argv) < 2:
        print("usage: python", sys.argv[0], "<source dir>")
        exit(1)

    files, words, freq_mat = read_input(sys.argv[1])
    tf_idf_mat, idf_vec, doc_vec_lens = calc_tf_idf_mat(freq_mat)
    print(
        query(
            "New new york",
            tf_idf_mat=tf_idf_mat,
            idf_vec=idf_vec,
            files=files,
            words=words,
            top_k=2,
            doc_vec_lens=doc_vec_lens))


if __name__ == "__main__":
    _test()
