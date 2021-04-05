import sys
import colorama
from math import log, floor, ceil
"""
This program implements the hamming codification algorithm.
	1) Bit stuffing in all 2^n, 0 <= n <= log_2(len(message))
	2) Xor all bits in data positions (complement of 2^n positions)
	3) The result is the hamming bits in reverse position
"""


def hammingCode(data):
    if not data:
        raise ValueError('Data must not be null!')

    coddedMessage = data

    j = 1
    i = 1
    k = 1
    stuffSeq = 0
    stuffLen = 0
    while j <= len(data):
        if k != i:
            if data[j - 1] == '1':
                stuffSeq = stuffSeq ^ k
            j += 1
        else:
            i *= 2
        k += 1
    gen = (c for c in bin(stuffSeq)[2:][::-1])
    i = 1
    while i <= len(coddedMessage):
        newVal = '0'
        try:
            newVal = gen.__next__()
        except:
            pass
        coddedMessage = coddedMessage[:(i - 1)] + newVal + coddedMessage[
            (i - 1):]
        i *= 2

    return coddedMessage


def printMessage(codifiedData):
    nxtVal = 1
    for i in range(1, len(codifiedData) + 1):
        if i == nxtVal:
            print(
                colorama.Fore.YELLOW + codifiedData[i - 1] +
                colorama.Fore.RESET,
                end='')
            nxtVal *= 2
        else:
            print(codifiedData[i - 1], end='')
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:', sys.argv[0], '<binary sequence to codify>')
        exit(1)
    printMessage(hammingCode(sys.argv[1]))
