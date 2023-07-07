# MADE_project_2023
Репозиторий для группового проекта построения модели генерации фейкдокументов.

**Краткое описание проекта:**  
Фейковые, сгенерированные нейросетями документы все чаще встречаются на просторах сети, прогнозируется их активный рост. В связи с этим, поисковые системы должны иметь устойчивость к подобного рода текстам, картинкам, видео и другим файлам, чтобы уберечь пользователей от искусственно созданной информации. Устойчивость можно проверить на практике, создав модель, генерирующую фейковые документы. Область изучения здесь открыта, хотя есть несколько удачных статей.
Задача проекта - создать модель генерации основанную на Transformer подобных архитектурах, которая может быть завернута в Python библиотеку с возможностью запуска основного функционала (в том числе генерации текстов) из командной строки. При желании можно оформить в виде сервиса с веб-интерфейсом, который принимает запрос и генерирует простую html страницу с текстом документа. Важным критерием будет то, насколько успешно сгенерированные документы смогут обмануть систему: насколько они хорошо ранжируются, являются правдоподобными.

  
  
**Полезные ссылки:**  

1. [Transformers Doc](https://huggingface.co/docs/transformers/performance)
2. [Введение в Transformers и Hugging Face](https://habr.com/ru/articles/704592/)
3. [Оценка выпускного проекта](https://data.vk.company/blog/topic/view/21655/)
4. [Тюнинг GPT-like моделей](https://habr.com/ru/companies/neoflex/articles/722584/)

**Инструкция по запуску:**
1. Склонировать репозиторий и установить зависимости.
2. Перейти в папку `/scripts` и запустить генерацию текстов:
~~~
python gen_text_rugpt.py -i test_samples_4.json -m train -o experiment_0_rugpt_data_for_metrics_4_samples.json -n rugpt
~~~
3. Далее запускаем команду, чтобы преобразовать данные в нужный формат для модели ранжирования:
~~~
python get_data_for_ranker.py -i experiment_0_rugpt_data_for_metrics_4_samples.json -o experiment_0_rugpt_4_samples.tsv
~~~
4. Запускаем ранжировщик:
~~~
cat experiment_0_rugpt_4_samples.tsv | cut -f2,4 | PYTHONPATH="$PYTHONPATH:/mnt/DATA/n.ermolaev/made/ranking-pipeline" python
 -m inferencer.inference --model inferencer/models/web/xlm_roberta_large_assessor_body_q40_b440 --gpus 4 --bs 64
 --model_data_path /mnt/DATA/n.ermolaev/made/models/xlm_roberta_large_assessor_body_q40_b440 --no_half > experiment_0_scores_m3_rugpt.txt
~~~
5. Считаем метрики:
~~~
python count_metric_script.py -i experiment_0_rugpt_data_for_metrics_4_samples.json -s experiment_0_scores_m3_rugpt.txt -o metrics_rugpt.json
~~~



