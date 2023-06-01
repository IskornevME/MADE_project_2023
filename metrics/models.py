# import tensorflow_hub as hub
# from tensorflow_text import SentencepieceTokenizer
# import tensorflow as tf
# from typing import List, Dict, Tuple, Union
#
#
# class ModelUSE():
#     """
#     Класс модели universal-sentence-encoder-multilingual-large-3,
#     реализующий загрузку модели и векторизацию текстов
#
#     Ссылка: https://huggingface.co/vprelovac/universal-sentence-encoder-multilingual-large-3
#     """
#
#     def __init__(self):
#         # Ограничение потребляемой памяти моделью
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                                 [tf.config.experimental.VirtualDeviceConfiguration(
#                                                                     memory_limit=1024)])
#         self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
#
#     def encode(self, sentences: List[str]):
#         """
#         Функция для векторизации списка текстов
#
#         Parameters
#         ------------
#         sentences: `List[str]`
#             Список  текстов
#
#         Returns
#         ------------
#         `List[List[int]]`
#             Список векторизированных текстов
#         """
#         embeddings = self.model(sentences)
#         return embeddings
