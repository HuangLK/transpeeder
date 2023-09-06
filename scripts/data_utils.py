from lingua import Language, LanguageDetectorBuilder


class LinguaLid:
    """
    https://github.com/pemistahl/lingua-py
    """
    def __init__(self):
        super().__init__()
        self.detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def detect(self, text: str):
        result = self.detector.detect_language_of(text)
        if result is None:
            return None
        else:
            return result.iso_code_639_1.name.lower()


if __name__ == '__main__':
    lid = LinguaLid()
    print(lid.detect('中の文.'))
    print(lid.detect('123'))
    print(lid.detect('NHANES III数据库中的期望死亡率可以通过使用Cox比例风险回归模型来确定。'))
    print(lid.detect('HistAuGAN的代码和模型可以在https://github.com/sophiajw/HistAuGAN上公开获取。'))