# План на 2 недели

Код уже написан и покрыт тестами. Твоя задача: запустить пайплайн на маке, получить реальные
цифры, разобраться в каждом модуле (на собеседовании спросят) и довести упаковку до уровня,
который видно с первого экрана GitHub.

**Правило:** коммить маленькими шагами каждый день. История из 40 осмысленных коммитов за две
недели выглядит намного лучше, чем один коммит «initial».

## Неделя 1 — данные и модели

| День | Что сделать | Результат |
|---|---|---|
| 1 | `git init`, `make setup`, `make test`. Прочитать `text.py` и тесты, понять транслитерацию | Зелёный CI на GitHub |
| 2 | `make data`. Открыть `stats.json`: сколько дубликатов между train и test, баланс классов | Цифры для раздела Data в README |
| 3 | Ноутбук `notebooks/01_eda.ipynb`: длины текстов, распределение классов, частые слова по темам | 2–3 графика для README |
| 4 | `make baseline`. Посмотреть `reports/baseline/errors.csv`: где модель ошибается и почему | Первая строка таблицы результатов |
| 5 | `make smoke`: проверить, что обучение на MPS идёт без ошибок. Замерить скорость (it/s) | Понимание, сколько займёт полный прогон |
| 6–7 | `make train` (лучше запускать на ночь). Если долго, уменьшить `max_train_samples` до 30–40k | `models/transformer/final` |

## Неделя 2 — продакшн и упаковка

| День | Что сделать | Результат |
|---|---|---|
| 8 | `make export` и `make eval`. Проверить `export_report.json`: совпадение INT8 с PyTorch должно быть ≥ 99% | Строки ONNX в таблице |
| 9 | Анализ ошибок: топ-путаницы из `metrics.json`, 10 примеров из `errors.csv`. Написать 3–4 вывода | Раздел Error analysis в README |
| 10 | `make serve` и `make docker`: проверить API через `/docs` | Рабочий Docker-образ |
| 11 | Залить модель на HF Hub (`hf upload <user>/mavzu models/onnx`) с model card (`docs/MODEL_CARD.md`) | Ссылка на модель |
| 12 | `make space` и залить Space. Записать GIF демо (латиница + кириллица) | Живое демо по ссылке |
| 13 | Заполнить все `TBD` и `TODO` в README, вставить картинки, поменять `<your-username>` | README готов |
| 14 | Пост в LinkedIn/Telegram с GIF и одной главной цифрой. Закрепить репо в профиле GitHub | Проект виден рекрутерам |

## Если останется время (по ценности)

1. **Реальная кириллица.** Собрать 200–300 кириллических новостей (например, с сайтов, которые
   публикуют обе версии), разметить их и проверить модель на настоящих данных, а не на
   транслитерированных. Это самый сильный пункт для резюме.
2. **tahrirchi-bert-base** (110M) на Kaggle GPU: сравнить с small по качеству и латентности.
3. **Метрики сервиса**: эндпоинт `/metrics` для Prometheus (число запросов, латентность).
4. **Объяснимость**: подсветка слов, которые повлияли на решение (attention или SHAP для baseline).

## Строки для резюме (заполни цифрами после прогона)

> **Mavzu — Uzbek news topic classifier** · Python, PyTorch, Transformers, ONNX, FastAPI, Docker
> - Fine-tuned an Uzbek BERT (TahrirchiBERT, 67M) on 500k-article news corpus for 15-class topic classification, reaching **X macro-F1** vs **Y** for a TF-IDF + SVM baseline.
> - Built a rule-based Cyrillic↔Latin transliteration layer; restored macro-F1 on Cyrillic input from **A** to **B**.
> - Found and removed **N** train/test near-duplicates (data leakage) in the public benchmark.
> - Exported to INT8 ONNX: **K× smaller**, **p50 latency L ms** on CPU, **≥99%** prediction agreement with PyTorch; served via FastAPI + Docker with CI and tests.

## Что спросят на собеседовании (будь готов ответить)

- Почему macro-F1, а не accuracy?
- Зачем char n-граммы в baseline для узбекского языка?
- Что такое утечка данных и как ты её нашёл?
- Почему fp32 на MPS, а не fp16?
- Что делает динамическая INT8-квантизация и почему качество почти не падает?
- Как бы ты масштабировал API до 1000 RPS?
