import numpy as np

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Union
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from .evaluation_metrics import (
    TopK, AverageLoc, FDARO, UpQuartile, AverageRelLoc
)
# from .models import ModelUSE
import torch


class Bm25:
    """Класс метрики ранжирования bm25"""

    def __init__(self):
        pass

    def name(self):
        return "Bm25"

    def ranking(self, query: str, sentences: List[str], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция ранжирования bm25. https://ru.wikipedia.org/wiki/Okapi_BM25

        Parameters
        ------------
        query: `str`
            Список токенов запроса
        sentences: `List[str]`
            Список списков токенов текстов
        labels: `List[int]`
            Список меток текстов

        Returns
        ------------
        `List[Tuple[float, int]]`
            Список оценок релевантности текстов
        """
        tokenized_query = self._encode(query)[0]
        tokenized_sentences = self._encode(sentences)
        bm25 = BM25Okapi(tokenized_sentences)
        scores = bm25.get_scores(tokenized_query)

        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: List[float], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция сортировки оценки и лейблов

        Parameters
        ------------
        scores: `List[float]`
            Массив оценок ранга присвоенных ранкером
        labels: `List[int]`
            Массив меток

        Returns
        ------------
        `List[List]`
            Отсортированный список ранжируемых элементов по релевантности
        """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)

    def _encode(self, sentences: Union[str, List[str]]) -> List[List[str]]:
        """
        Функция для декодирования предложений в последовательность токенов

        Parameters
        ------------
        sentences: `str, List[str]`
            Строка или массив строк для разбиения на токены

        Returns
        ------------
        `List[List[str]]`
            Последовательность токенов
        """
        # tokenized_sentences = self.tokenizer.encode(sentences, return_tensors="pt")
        # TODO: Сделать более умную токенизацию
        if isinstance(sentences, str):
            sentences = [sentences]

        tokenized_sentences = []
        for cur_sent in sentences:
            tokenized_sentences.append(cur_sent.split(" "))
        return tokenized_sentences


class LaBSE:
    """Класс метрики ранжирования LaBSE"""

    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/LaBSE')

    def name(self) -> str:
        return "LaBSE"

    def ranking(self, query: str, sentences: List[str], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция ранжирования LaBSE

        Parameters
        ------------
        query: `str`
            Строка запроса
        sentences: `List[str]`
            Список строк текстов
        labels: `List[int]`
            Список меток текстов

        Returns
        ------------
        `List[Tuple[float, int]]`
            Список оценок релевантности текстов
        """
        query = self.model.encode(query)
        embeddings = self.model.encode(sentences)
        scores = util.pytorch_cos_sim(query, embeddings).numpy()[0]
        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: List[float], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция сортировки оценки и лейблов

        Parameters
        ------------
        scores: `List[float]`
            Массив оценок ранка присвоенных ранкером
        labels: `List[int]`
            Массив меток

        Returns
        ------------
        `List[Tuple[float, int]]`
            Отсортированный список ранжируемых элементов по релевантности
        """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)


class MsMarcoST:
    """Класс метрики ранжирования MS MARCO из sentence-transformers"""

    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

    def name(self) -> str:
        return "MsMarcoST"

    def ranking(self, query: str, sentences: List[str], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция ранжирования MsMarcoST

        Parameters
        ------------
        query: `str`
            Строка запроса
        sentences: `List[str]`
            Список строк текстов
        labels: `List[int]`
            Список меток текстов

        Returns
        ------------
        `List[Tuple[float, int]]`
            Список пар, где пара имеет вид (скор, метка), отсортированных по убыванию скоров
        """
        query = self.model.encode(query)
        embeddings = self.model.encode(sentences)
        scores = util.dot_score(query, embeddings).numpy()[0]
        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: List[float], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция сортировки оценки и лейблов

        Parameters
        ------------
        scores: `List[float]`
            Массив оценок ранка присвоенных ранкером
        labels: `List[int]`
            Массив меток

        Returns
        ------------
        `List[Tuple[float, int]]`
            Отсортированный список ранжируемых элементов по релевантности
        """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)


class MsMarcoCE:
    """
    Класс метрики ранжирования MS MARCO из cross-encoder.
    Данная метрика более устойчива к кейсам, когда пассаж полностью повторяет запрос.
    """

    def __init__(self) -> None:
        self.model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)

    def name(self) -> str:
        return "MsMarcoCE"

    def ranking(self, query: str, sentences: List[str], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция ранжирования MsMarcoCE

        Parameters
        ------------
        query: `str`
            Строка запроса
        sentences: `List[str]`
            Список строк текстов
        labels: `List[int]`
            Список меток текстов

        Returns
        ------------
        `List[Tuple[float, int]]`
            Список пар, где пара имеет вид (скор, метка), отсортированных по убыванию скоров
        """

        pairs_que_sent = []
        for sent in sentences:
            pairs_que_sent.append((query, sent))
        scores = self.model.predict(pairs_que_sent)
        scores = self._sorted(scores, labels)
        return scores

    def _sorted(self, scores: List[float], labels: List[int]) -> List[Tuple[float, int]]:
        """
        Функция сортировки оценки и лейблов

        Parameters
        ------------
        scores: `List[float]`
            Массив оценок ранка присвоенных ранкером
        labels: `List[int]`
            Массив меток

        Returns
        ------------
        `List[Tuple[float, int]]`
            Отсортированный список ранжируемых элементов по релевантности
        """
        return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)

# class USE:
#     """Класс метрики ранжирования USE"""
#     def __init__(self):
#         self.model = ModelUSE()
#
#     def name(self) -> str:
#         return "USE"
#
#     def ranking(self, query: str, sentences: List[str], labels: List[int]) -> List[Tuple[float, int]]:
#         """
#         Функция ранжирования USE
#
#         Parameters
#         ------------
#         query: `str`
#             Строка запроса
#         sentences: `List[str]`
#             Список строк текстов
#         labels: `List[int]`
#             Список меток текстов
#
#         Returns
#         ------------
#         `List[Tuple[float, int]]`
#             Список оценок релевантности текстов
#         """
#
#         sentences.append(query)
#         embeddings = [x.numpy() for x in self.model.encode(sentences)]
#         embeddings = torch.Tensor(embeddings)
#         scores = util.pytorch_cos_sim(embeddings[-1], embeddings[:-1]).numpy()[0]
#         scores = self._sorted(scores, labels)
#         return scores
#
#     def _sorted(self, scores: List[float], labels: List[int]) -> List[Tuple[float, int]]:
#         """
#         Функция сортировки оценки и лейблов
#
#         Parameters
#         ------------
#         scores: `List[float]`
#             Массив оценок ранка присвоенных ранкером
#         labels: `List[int]`
#             Массив меток
#
#         Returns
#         ------------
#         `List[Tuple[float, int]]`
#             Отсортированный список ранжируемых элементов по релевантности
#         """
#         return sorted([item for item in zip(scores, labels)], key=lambda x: x[0], reverse=True)


class RankingMetrics:
    """Класс аккумулирующий все метрики"""
    FAKE_DOC_LABEL: int = -1

    def __init__(self, metrics, relevant_doc_labels: Union[int, List] = 1) -> None:
        """

        Parameters
        ------------
        metrics: `Union[LaBSE, USE, Bm25, MsMarcoCE, MsMarcoST]`
            Классы метрик ранжирования

        relevant_doc_label: `Union[int, List]`
            Метка или массив меток, обозначающих релевантные документы. (Используется для оценки FDARO)
        """
        # Среднее место фейковых документов в финальной выдаче
        self.average_place_fake_doc = AverageLoc(metrics)
        # Количество случаев когда фейковый документ выше релевантного
        self.fake_doc_above_relevant_one = FDARO(metrics, relevant_doc_labels)
        # Количество случаев когда фейковый документ вошел в топ 1
        self.fake_top_k = TopK(metrics)
        self.upper_quartile = UpQuartile(metrics)
        # Среднее относительное место фейковых документов в выдаче
        self.average_rel_place_fake_doc = AverageRelLoc(metrics)
        # Массив метрик для подсчета
        self.metrics = metrics
        # Число моделей для подсчета метрик
        self.num_metrics = len(metrics)

    def update(self, query: str, sentences: List[str], labels: List[int]) -> None:
        """
       Функция обновления всех метрик по переданным данным

       Parameters
       ------------
       query: `str`
           Строка запроса
       sentences: `List[str]`
           Список строк текстов
       labels: `List[int]`
           Список меток текстов
       """
        if not isinstance(query, str):
            raise TypeError("The request must be a string!")

        if len(sentences) != len(labels):
            raise ValueError("len(labels) must be equal to len(sentences)")

        for item in sentences:
            if not isinstance(item, str):
                raise TypeError("The sentences must be of the `List[str]` type!")

        for item_label in labels:
            if not isinstance(item_label, int):
                raise TypeError("The labels must be of the `List[int]` type!")

        for cur_metric in self.metrics:
            ranking_list = cur_metric.ranking(query, sentences, labels)
            self.fake_top_k.update(cur_metric.name(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.fake_doc_above_relevant_one.update(cur_metric.name(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.average_place_fake_doc.update(cur_metric.name(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.average_rel_place_fake_doc.update(cur_metric.name(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)
            self.upper_quartile.update(cur_metric.name(), ranking_list, RankingMetrics.FAKE_DOC_LABEL)

    def get(self) -> Dict:
        """
        Функция для получения значения всех метрик

        Returns
        ----------
        `Dict`
            Словарь значений метрик
        """
        result = {}
        for key_, value in self.average_place_fake_doc.get().items():
            result[key_] = value

        for key_, value in self.fake_top_k.get().items():
            result[key_] = value

        for key_, value in self.fake_doc_above_relevant_one.get().items():
            result[key_] = value

        for key_, value in self.upper_quartile.get().items():
            result[key_] = value

        for key_, value in self.average_rel_place_fake_doc.get().items():
            result[key_] = value

        return result

    def show_metrics(self) -> None:
        for i, (key_, value) in enumerate(self.average_place_fake_doc.get().items()):
            print(f"{key_}: {np.round(value, 2)}", end="   ")

        print("\n-----------------------------")
        for i, (key_, value) in enumerate(self.average_rel_place_fake_doc.get().items()):
            print(f"{key_}: {np.round(value, 2)}", end="   ")

        print("\n-----------------------------")
        fake_top_k_it = list(self.fake_top_k.get().items())
        for i in range(3):
            for j in range(self.num_metrics):
                print(f"{fake_top_k_it[i + 3*j][0]}: {np.round(fake_top_k_it[i + 3*j][1], 2)}", end="   ")
            print()

        print("-----------------------------")
        fake_doc_above_relevant_one_it = list(self.fake_doc_above_relevant_one.get().items())
        for i in range(2):
            for j in range(self.num_metrics):
                print(f"{fake_doc_above_relevant_one_it[i + 2*j][0]}: {np.round(fake_doc_above_relevant_one_it[i + 2*j][1], 4)}", end="   ")
            print()

        print("-----------------------------")
        for i, (key_, value) in enumerate(self.upper_quartile.get().items()):
            print(f"{key_}: {np.round(value, 2)}", end="   ")
        print("\n\n")
