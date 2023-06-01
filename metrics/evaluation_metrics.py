from typing import List, Union, Tuple, Dict
import numpy as np


class TopK:
    TOP1 = 1
    TOP3 = 3
    TOP5 = 5

    def __init__(self, ranking_metrics: List) -> None:
        self.metrics = {}
        self.calls_cnt = {}
        self._separator = "_"
        self._top_numbers = [TopK.TOP1, TopK.TOP3, TopK.TOP5]
        for cur_metrics in ranking_metrics:
            for cur_top in self._top_numbers:
                self.metrics[cur_metrics.name() + self._separator + self.name() + str(cur_top)] = 0
                self.calls_cnt[cur_metrics.name() + self._separator + self.name() + str(cur_top)] = 0

    def update(self, metric_name: str,
               ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]],
               fake_doc_label: Union[int, List[int]] = -1) -> None:
        """
        Функция для обновления значений метрики

        Parameters
        -------------
        metric_name: `str`
            Название метрики
        ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
            Результат ранжирующей модели
        fake_doc_label: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих фейковым документам
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        _check_update_args(fake_doc_label, ranking_list)

        for top_num in self._top_numbers:
            for item in ranking_list[:min(len(ranking_list), top_num)]:
                if item[1] in fake_doc_label:
                    self.metrics[metric_name + self._separator + self.name() + str(top_num)] += 1

            self.calls_cnt[metric_name + self._separator + self.name() + str(top_num)] += 1

    def get(self) -> Dict[str, float]:
        """
        Функция для получения словаря значений подсчитанных метрик

        Returns
        -------------
        `Dict[str, float]`
            Словарь подсчитанных значений метрик
        """
        result = {}
        for metric_name, value in self.metrics.items():
            metric_name_ = metric_name.split("_")[0]
            top_num = int(metric_name.split('@')[1])
            result[metric_name_ + self._separator + self.name() + str(top_num)] = \
                value / max(1, self.calls_cnt[metric_name_ + self._separator + self.name() + str(top_num)])

        return result

    def name(self) -> str:
        '''
        Функция для получения имени метрики

        Returns
        -------------
        `str`
            Имя метрики
        '''
        return "Top@"


class FDARO:
    """
    Метрика оценивает как часто фейковый документ находится выше релевантного.
    Представлены две версии. Первая (FDARO@v1) оценивает как часто фейк документ
    находится выше ВСЕХ релевантных. Вторая (FDARO@v2) - как часто фейковый документ
    оказывается выше хотя бы одного.
    """

    def __init__(self, ranking_metrics: List, relevant_doc_labelss: Union[int, List] = 1) -> None:
        self._separator = "_"
        self.metrics, self.calls_cnt = {}, {}
        self.versions = ["v1", "v2"]
        if isinstance(relevant_doc_labelss, int):
            self.relevant_doc_labelss = [relevant_doc_labelss]
        else:
            self.relevant_doc_labelss = relevant_doc_labelss

        for cur_metric in ranking_metrics:
            for version in self.versions:
                self.metrics[cur_metric.name() + self._separator + self.name() + version] = 0
                self.calls_cnt[cur_metric.name() + self._separator + self.name() + version] = 0

    def update(self, metric_name: str,
               ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]],
               fake_doc_label: Union[int, List] = -1) -> None:
        """
        Функция для обновления значений метрики

        Parameters
        -------------
        metric_name: `str`
            Название метрики
        ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
            Результат ранжирующей модели
        fake_doc_label: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих фейковым документам
        relevant_doc_labelss: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих релевантным документам
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        _check_update_args(fake_doc_label, ranking_list)

        is_first = False
        for item in ranking_list:
            if item[1] in self.relevant_doc_labelss:
                break
            elif item[1] in fake_doc_label:
                is_first = True
                break

        if is_first:
            self.metrics[metric_name + self._separator + self.name() + self.versions[0]] += 1
        self.calls_cnt[metric_name + self._separator + self.name() + self.versions[0]] += 1

        scores = []
        selected = []
        for item in ranking_list:
            scores.append(item[0])
            selected.append(item[1])

        scores_relevant = -1e9
        scores_fake = -1e9
        for ind in range(len(selected)):
            # Выбираем последний релевантный элемент
            if selected[ind] in self.relevant_doc_labelss:
                scores_relevant = scores[ind]
            elif selected[ind] in fake_doc_label and scores_fake == -1e9:
                scores_fake = scores[ind]

        upper_or_not = (scores_fake - scores_relevant) > 1e-12
        self.metrics[metric_name + self._separator + self.name() + self.versions[1]] += int(upper_or_not)
        self.calls_cnt[metric_name + self._separator + self.name() + self.versions[1]] += 1

    def get(self) -> Dict[str, float]:
        """
        Функция для получения словаря значений подсчитанных метрик

        Returns
        -------------
        `Dict[str, float]`
            Словарь подсчитанных значений метрик
        """
        result = {}
        for metric_name, value in self.metrics.items():
            version = metric_name.split('@')[1]
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.name() + version] = \
                value / max(1, self.calls_cnt[metric_name + self._separator + self.name() + version])

        return result

    def name(self) -> str:
        '''
        Функция для получения имени метрики

        Returns
        -------------
        `str`
            Имя метрики
        '''
        return "FDARO@"


class AverageLoc:
    """Метрика для оценки среднего места фейкового документа"""

    def __init__(self, ranking_metrics) -> None:
        self._separator = "_"
        self.metrics, self.calls_cnt = {}, {}
        for cur_metric in ranking_metrics:
            self.metrics[cur_metric.name() + self._separator + self.name()] = 0
            self.calls_cnt[cur_metric.name() + self._separator + self.name()] = 0

    def update(self, metric_name: str,
               ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]],
               fake_doc_label: Union[int, list] = -1) -> None:
        """
        Функция для обновления значений метрики

        Parameters
        -------------
        metric_name: `str`
            Название метрики
        ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
            Результат ранжирующей модели
        fake_doc_label: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих фейковым документам
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        _check_update_args(fake_doc_label, ranking_list)

        is_fake = False
        for ind, item in enumerate(ranking_list):
            if item[1] in fake_doc_label:
                self.metrics[metric_name + self._separator + self.name()] += (ind + 1)
                is_fake = True
        # Если нет фейковых документов в рейтинге, предполагаем что он идет после всех т.е последним
        if not is_fake:
            self.metrics[metric_name + self._separator + self.name()] += len(ranking_list) + 1

        self.calls_cnt[metric_name + self._separator + self.name()] += 1

    def get(self) -> Dict[str, float]:
        """
        Функция для получения словаря значений подсчитанных метрик

        Returns
        -------------
        `Dict[str, float]`
            Словарь подсчитанных значений метрик
        """
        result = {}
        for metric_name, value in self.metrics.items():
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.name()] = \
                value / max(1, self.calls_cnt[metric_name + self._separator + self.name()])

        return result

    def name(self) -> str:
        '''
        Функция для получения имени метрики

        Returns
        -------------
        `str`
            Имя метрики
        '''
        return "AverageLoc"


class AverageRelLoc:
    """
    Метрика для оценки среднего относительного места фейкового документа.
    Чем ближе число к нулю, тем выше ставится фейковый документ в ранжировании
    """

    def __init__(self, ranking_metrics) -> None:
        self._separator = "_"
        self.metrics, self.calls_cnt = {}, {}
        for cur_metric in ranking_metrics:
            self.metrics[cur_metric.name() + self._separator + self.name()] = 0  # инициализируем значение метрики
            self.calls_cnt[cur_metric.name() + self._separator + self.name()] = 0

    def update(self, metric_name: str,
               ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]],
               fake_doc_label: Union[int, list] = -1) -> None:
        """
        Функция для обновления значений метрики

        Parameters
        -------------
        metric_name: `str`
            Название метрики
        ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
            Результат ранжирующей модели
        fake_doc_label: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих фейковым документам
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        _check_update_args(fake_doc_label, ranking_list)

        is_fake = False
        for ind, item in enumerate(ranking_list):
            if item[1] in fake_doc_label:
                self.metrics[metric_name + self._separator + self.name()] += (ind + 1) / len(ranking_list)
                is_fake = True

        if not is_fake:
            self.metrics[metric_name + self._separator + self.name()] += 1

        self.calls_cnt[metric_name + self._separator + self.name()] += 1

    def get(self) -> Dict[str, float]:
        """
        Функция для получения словаря значений подсчитанных метрик

        Returns
        -------------
        `Dict[str, float]`
            Словарь подсчитанных значений метрик
        """
        result = {}
        for metric_name, value in self.metrics.items():
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.name()] = \
                value / max(1, self.calls_cnt[metric_name + self._separator + self.name()])

        return result

    def name(self) -> str:
        '''
        Функция для получения имени метрики

        Returns
        -------------
        `str`
            Имя метрики
        '''
        return "AverageRelLoc"


class UpQuartile:
    """Метрика для оценки частоты попадания фейкового документа в топ 25%"""

    def __init__(self, ranking_metrics) -> None:
        self._separator = "_"
        self.metrics, self.calls_cnt = {}, {}
        for cur_metric in ranking_metrics:
            self.metrics[cur_metric.name() + self._separator + self.name()] = 0
            self.calls_cnt[cur_metric.name() + self._separator + self.name()] = 0

    def update(self, metric_name: str,
               ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]],
               fake_doc_label: Union[int, list] = -1) -> None:
        """
        Функция для обновления значений метрики

        Parameters
        -------------
        metric_name: `str`
            Название метрики
        ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
            Результат ранжирующей модели
        fake_doc_label: `Union[int, List[int]]`
            Метка или массив меток, принадлежащих фейковым документам
        """
        if isinstance(fake_doc_label, int):
            fake_doc_label = [fake_doc_label]

        _check_update_args(fake_doc_label, ranking_list)

        scores = []
        selected = []
        for item in ranking_list:
            scores.append(item[0])
            selected.append(item[1])
        selected = np.array(selected)
        scores = np.array(scores)

        upper_quartile = np.quantile(scores, 0.75)

        scores_fake = -1e9
        for ind in range(len(selected)):
            if selected[ind] in fake_doc_label:
                scores_fake = selected[ind]
                break

        if scores_fake >= upper_quartile:
            self.metrics[metric_name + self._separator + self.name()] += 1

        self.calls_cnt[metric_name + self._separator + self.name()] += 1

    def get(self) -> Dict[str, float]:
        """
        Функция для получения словаря значений подсчитанных метрик

        Returns
        -------------
        `Dict[str, float]`
            Словарь подсчитанных значений метрик
        """
        result = {}
        for metric_name, value in self.metrics.items():
            metric_name = metric_name.split("_")[0]
            result[metric_name + self._separator + self.name()] = \
                value / max(1, self.calls_cnt[metric_name + self._separator + self.name()])

        return result

    def name(self) -> str:
        '''
        Функция для получения имени метрики

        Returns
        -------------
        `str`
            Имя метрики
        '''
        return "UpQuartile"


def _check_update_args(fake_doc_label: Union[int, List[int]],
                       ranking_list: List[Union[Tuple[float, int], List[Union[float, int]]]]) -> None:
    '''
    Функция для валидации списка меток и списка для ранжирования

    Parameters
    -------------
    fake_doc_label: `Union[int, List[int]]`
        Метка или список меток, обозначающих фейковый документ
    ranking_list: `List[Union[Tuple[float, int], List[float, int]]]`
        Список оценок ранжировщика
    '''
    if not isinstance(fake_doc_label, list):
        raise TypeError("The tags of fake documents must be 'int' or 'list!'")
    else:
        for label in fake_doc_label:
            if not isinstance(label, int):
                raise TypeError("The tags of fake documents must be 'int'!")

    if not isinstance(ranking_list, list):
        raise TypeError("The list of ranked elements should be of type 'list'!")
    else:
        for item in ranking_list:
            if not isinstance(item, (tuple, list)) or \
                    not isinstance(item[0], (np.floating, float)) or not isinstance(item[1], int) or \
                    len(item) != 2:
                raise TypeError("The list of ranked elements should contain pairs of elements: rating, label")
