class SubWord(object):
    @staticmethod
    def strB2Q(words):
        s = []
        for w in words:
            code = ord(w)
            if code == 0x3000:
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            s.append(chr(code))
        return ''.join(s)

    @staticmethod
    def subword(words):
        words = SubWord.strB2Q(words)

        # 只针对非中文, only for not chinese string
        if SubWord.is_cn(words):
            return words

        if '年' in words and SubWord.has_num(words):
            return '_YEAR'

        if '月' in words and SubWord.has_num(words):
            return '_MONTH'

        if ('日' in words or '号' in words) and SubWord.has_num(words):
            return '_DAY'

        if ('分之' in words or '%' in words) and SubWord.has_num(words):
            return '_PERCENT'

        if 'www' in words or 'http:' in words:
            return '_NET'

        if ('·' in words or '.' in words or '*' in words) and len(words) > 1 and not SubWord.has_num(
                words):
            return '_NAME'

        if '-' in words or '、' in words:
            parts = words.split('-')
            if len(parts) >= 2:
                for part in parts:
                    if SubWord.is_cn(part) or SubWord.is_eng(part):
                        continue
                    else:
                        break
                return '_NAME'

        if ('时' in words or '分' in words) and SubWord.has_num(words):
            return '_TIME'

        if ('百' in words or '千' in words or '万' in words or '亿' in words or '兆') and SubWord.has_num(words):
            return '_NUMBER'

        # if words.find('点') != -1:
        #     if words.find("一") != -1 or words.find("二") != -1 or words.find("三") != -1 or words.find("四") != -1 or \
        #             words.find("五") != -1 or words.find("六") != -1 or words.find("七") != -1 or \
        #             words.find("八") != -1 or words.find("九") != -1 or words.find("十") != -1:
        #         return '_NUMBER'

        if SubWord.is_eng(words):
            return '_ENG'

        return words

    @staticmethod
    def has_num(words):
        if '零' in words or '一' in words or '二' in words or '三' in words or '四' in words or '五' in words \
                or '六' in words or '七' in words or '八' in words or '九' in words or '十' in words \
                or '0' in words or '1' in words or '2' in words or '3' in words or '4' in words or '5' in words \
                or '6' in words or '7' in words or '8' in words or '9' in words:
            return True
        return False

    @staticmethod
    def is_eng(words):
        for w in words:
            if ('\uFF21' <= w <= '\uFF3A') or ('\uFF41' <= w <= '\uFF5A') \
                    or (u'\u0041' <= w <= u'\u005a') or (u'\u0061' <= w <= u'\u007a'):
                continue
            else:
                return False
        return True

    @staticmethod
    def is_cn(words):
        for w in words:
            if '\u4e00' <= w <= '\u9fa5':
                continue
            else:
                return False
        return True


if __name__ == '__main__':
    with open('data/special.txt', 'r', encoding='utf-8') as f_in, \
            open('data/special_after.txt', 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line == SubWord.subword(line):
                f_out.write(line)
                f_out.write('\n')
