cur_mask = int(b"11111000", base=2)
MASKS = []

while cur_mask:
    MASKS.insert(0, cur_mask)
    cur_mask <<= 1
    cur_mask &= 255


CORRUPTED_EXCEP = "Can't decode stream."


def _decode_next_bytes(stream, start, size):
    if len(stream) < start + size:
        raise RuntimeError(CORRUPTED_EXCEP)

    codepoint = 0

    for i in range(size):
        b = stream[start + i]

        if b & MASKS[1] == MASKS[0]:
            codepoint += (b & ~MASKS[1]) << (size - i - 1) * 6

        else:
            raise RuntimeError(CORRUPTED_EXCEP)

    return codepoint


def decode(stream: str) -> str:
    i = 0
    codepoints = []

    while i < len(stream):
        b = stream[i]

        if b & MASKS[0] == 0:
            # Single-byte codepoint
            codepoints.append(b)
            i += 1

        else:
            for j in range(2, len(MASKS)):
                if b & MASKS[j] == MASKS[j - 1]:
                    base = (b & ~MASKS[j]) << (j - 1) * 6
                    codepoints.append(base + _decode_next_bytes(stream, i + 1, j - 1))
                    i += j
                    break

            else:
                raise RuntimeError(CORRUPTED_EXCEP)

    return codepoints


def _test():
    string = '''
    Source: https://en.wikipedia.org/wiki/Kanji#Ky%C5%8Diku_kanji

    The on'yomi (音読み, [oɰ̃jomi], lit. "sound(-based) reading"), the Sino-Japanese reading, is the modern descendant of the Japanese approximation of the base Chinese pronunciation of the character at the time it was introduced. It was often previously referred to as translation reading, as it was recreated readings of the Chinese pronunciation but was not the Chinese pronunciation or reading itself, similar to the English pronunciation of Latin loanwords. Old Japanese scripts often stated that on'yomi readings were also created by the Japanese during their arrival and re-borrowed by the Chinese as their own. There also exist kanji created by the Japanese and given an on'yomi reading despite not being a Chinese-derived or a Chinese-originating character. Some kanji were introduced from different parts of China at different times, and so have multiple on'yomi, and often multiple meanings. Kanji invented in Japan would not normally be expected to have on'yomi, but there are exceptions, such as the character 働 "to work", which has the kun'yomi "hataraku" and the on'yomi "dō", and 腺 "gland", which has only the on'yomi "sen"—in both cases these come from the on'yomi of the phonetic component, respectively 動 "dō" and 泉 "sen".
    
    Generally, on'yomi are classified into four types according to their region and time of origin:
    
        Go-on (呉音, "Wu sound") readings are from the pronunciation during the Northern and Southern dynasties of China during the 5th and 6th centuries. Go refers to the Wu region (in the vicinity of modern Shanghai), which still maintains linguistic similarities with modern Sino-Japanese vocabulary. See also: Wu Chinese and Shanghainese language.
        Kan-on (漢音, "Han sound") readings are from the pronunciation during the Tang dynasty of China in the 7th to 9th centuries, primarily from the standard speech of the capital, Chang'an (modern Xi'an). Here, Kan refers to Han Chinese people or China proper.
        Tō-on (唐音, "Tang sound") readings are from the pronunciations of later dynasties of China, such as the Song and Ming. They cover all readings adopted from the Heian era to the Edo period. This is also known as Tōsō-on (唐宋音, Tang and Song sound).
        Kan'yō-on (慣用音, "customary sound") readings, which are mistaken or changed readings of the kanji that have become accepted into the Japanese language. In some cases, they are the actual readings that accompanied the character's introduction to Japan, but do not match how the character "should" (is prescribed to) be read according to the rules of character construction and pronunciation.
    
    Examples (rare readings in parentheses) Kanji 	Meaning 	Go-on 	Kan-on 	Tō-on 	Kan'yō-on
    明 	bright 	myō 	mei 	(min) 	—
    行 	go 	gyō
    gō 	kō
    kō 	(an) 	—
    極 	extreme 	goku 	kyoku 	— 	—
    珠 	pearl 	shu 	shu 	ju 	(zu)
    度 	degree 	do 	(to) 	— 	—
    輸 	transport 	(shu) 	(shu) 	— 	yu
    雄 	masculine 	— 	— 	— 	yū
    熊 	bear 	— 	— 	— 	yū
    子 	child 	shi 	shi 	su 	—
    清 	clear 	shō 	sei 	(shin) 	—
    京 	capital 	kyō 	kei 	(kin) 	—
    兵 	soldier 	hyō 	hei 	— 	—
    強 	strong 	gō 	kyō 	— 	—
    
    The most common form of readings is the kan-on one, and use of a non-kan-on reading in a word where the kan-on reading is well known is a common cause of reading mistakes or difficulty, such as in ge-doku (解毒, detoxification, anti-poison) (go-on), where 解 is usually instead read as kai. The go-on readings are especially common in Buddhist terminology such as gokuraku (極楽, paradise), as well as in some of the earliest loans, such as the Sino-Japanese numbers. The tō-on readings occur in some later words, such as isu (椅子, chair), futon (布団, mattress), and andon (行灯, a kind of paper lantern). The go-on, kan-on, and tō-on readings are generally cognate (with rare exceptions of homographs; see below), having a common origin in Old Chinese, and hence form linguistic doublets or triplets, but they can differ significantly from each other and from modern Chinese pronunciation.
    
    In Chinese, most characters are associated with a single Chinese sound, though there are distinct literary and colloquial readings. However, some homographs (多音字 pinyin: duōyīnzì) such as 行 (háng or xíng) (Japanese: an, gō, gyō) have more than one reading in Chinese representing different meanings, which is reflected in the carryover to Japanese as well. Additionally, many Chinese syllables, especially those with an entering tone, did not fit the largely consonant-vowel (CV) phonotactics of classical Japanese. Thus most on'yomi are composed of two morae (beats), the second of which is either a lengthening of the vowel in the first mora (to ei, ō, or ū), the vowel i, or one of the syllables ku, ki, tsu, chi, fu (historically, later merged into ō and ū), or moraic n, chosen for their approximation to the final consonants of Middle Chinese. It may be that palatalized consonants before vowels other than i developed in Japanese as a result of Chinese borrowings, as they are virtually unknown in words of native Japanese origin, but are common in Chinese.
    
    On'yomi primarily occur in multi-kanji compound words (熟語, jukugo) words, many of which are the result of the adoption, along with the kanji themselves, of Chinese words for concepts that either did not exist in Japanese or could not be articulated as elegantly using native words. This borrowing process is often compared to the English borrowings from Latin, Greek, and Norman French, since Chinese-borrowed terms are often more specialized, or considered to sound more erudite or formal, than their native counterparts (occupying a higher linguistic register). The major exception to this rule is family names, in which the native kun'yomi are usually used (though on'yomi are found in many personal names, especially men's names). 
    '''
    print("Original string:", string)
    res = decode(string.encode("utf-8"))
    comb = "".join(map(chr, res))
    print("Result (in unicode codepoints):", res)
    print("Result (in unicode characters):", comb)
    assert comb == string


if __name__ == "__main__":
    _test()
