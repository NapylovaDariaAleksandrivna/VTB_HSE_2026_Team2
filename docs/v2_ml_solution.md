# v2 ML-решение кейса

Этот документ описывает отдельное решение для датасета из папки `v2`. Оно не заменяет первую Excel-версию, а дополняет ее: в v2 уже есть отрицательные H3-зоны, поэтому можно корректно обучить модель.

## Где находятся файлы

| Что | Файл |
|---|---|
| Входные транзакции | `data.parquet` |
| Target | `target.parquet` |
| Основной notebook | `notebooks/train_ml.ipynb` |
| CLI-скрипт | `scripts/analyze_case_v2_ml.py` |
| Все зоны со скорингом | `output/data/v2_zone_scores_ml.csv` |
| Полная витрина признаков | `output/data/v2_cell_features_full.csv` |
| Шорт-лист новых банкоматов | `output/data/v2_shortlist_new_atm.csv` |
| Зоны усиления сети | `output/data/v2_shortlist_capacity_existing_network.csv` |
| Метрики модели и диагностика | `output/data/v2_diagnostics.json` |
| Важность признаков | `output/data/v2_model_feature_importance.csv` |
| Карта | `output/maps/v2_atm_ml_potential_map.html` |
| Презентация | `output/pptx/vtb_atm_geoanalytics_solution_v2_ml.pptx` |
| Краткое резюме | `output/v2_solution_summary.md` |

## Почему в v2 уже можно обучать ML

В первой Excel-версии `target` совпадал с уникальными парами `h3_index + customer_id` из `data`, поэтому отрицательного класса не было.

В v2 ситуация другая:

- `data.parquet`: 4 151 096 строк;
- H3-зон в `data`: 8154;
- positive H3-зон с target-сигналом: 1654;
- negative H3-зон без target-сигнала: 6500.

Поэтому целевая переменная строится на уровне зоны:

```text
cell_y = 1, если h3_index есть в target.parquet
cell_y = 0, если h3_index есть в data.parquet, но отсутствует в target.parquet
```

Парная задача `customer_id + h3_index` не используется как основная, потому что часть target-пар отсутствует в `data` как транзакционная пара. Для задачи размещения банкомата H3-level постановка логичнее: банк выбирает зону, а не отдельную пару клиент-зона.

## Какие признаки считаются

### Спрос

- `unique_customers` - число уникальных клиентов в H3;
- `total_tx_count` - сумма `count`;
- `total_tx_sum` - сумма `sum`;
- `tx_per_customer` - операций на клиента;
- `sum_per_customer` - оборот на клиента;
- `avg_ticket_weighted` - средний чек зоны.

### Денежное поведение

- `max_transaction` - максимальная операция;
- `transaction_volatility` - агрегированная волатильность чеков;
- `small_ticket_share` - доля операций с малым средним чеком;
- `large_ticket_share` - доля операций с крупным средним чеком.

### MCC

- `unique_mcc_count` - число разных MCC;
- `mcc_entropy` - энтропия MCC;
- `top_mcc_share` - доля крупнейшего MCC;
- `mcc_vector_*` - доля каждого MCC в операциях зоны;
- `atm_affinity_mcc_score` - справочный target-derived MCC-сигнал.

`atm_affinity_mcc_score` не используется в обучении модели, чтобы не завышать качество через leakage.

### Время

- `active_time_buckets` - число активных временных бакетов;
- `time_entropy` - равномерность активности во времени;
- `peak_bucket_share` - доля самого активного бакета;
- `off_peak_activity` - активность вне пика.

### OSM и инфраструктура

Через Overpass API подтягиваются:

- банкоматы и отделения ВТБ;
- банкоматы и банки конкурентов;
- торговые центры;
- транспортные объекты;
- университеты и колледжи.

Из них считаются:

- `vtb_atm`, `vtb_branch`;
- `competitor_atm`, `competitor_bank`;
- `mall`, `transit`, `education`;
- `distance_to_nearest_vtb_atm`;
- `distance_to_nearest_competitor_atm`;
- `competitor_atm_count_500m`;
- `poi_count_500m`;
- `metro_distance`;
- `coverage_gap`.

## Модель

Используется `RandomForestClassifier`:

```text
n_estimators = 500
max_depth = 12
min_samples_leaf = 4
class_weight = balanced_subsample
```

Данные делятся на train/test через stratified holdout 75/25. Модель предсказывает:

```text
ml_atm_probability = P(cell_y = 1 | признаки H3)
```

Текущие метрики holdout:

- ROC-AUC: 0.887;
- PR-AUC: 0.725;
- Precision top 20%: 0.662;
- Recall top 20%: 0.652.

## Итоговый скоринг

ML-вероятность не используется напрямую как единственный ответ. Для бизнес-рекомендации строится итоговый индекс:

```text
placement_score = 100 * (
    0.55 * ml_atm_probability
  + 0.25 * demand_score_v2
  + 0.20 * coverage_gap_score
)
```

Смысл весов:

- `0.55` на ML - теперь есть валидный target с отрицательными зонами, поэтому модель становится главным сигналом;
- `0.25` на спрос - прямой транзакционный масштаб нужен, чтобы не выбирать зону только из-за похожести на target;
- `0.20` на gap покрытия - зона без найденного ВТБ-присутствия получает больший приоритет для новой точки.

## Как формируется шорт-лист

`v2_shortlist_new_atm.csv` выбирает зоны, где:

- высокий `placement_score`;
- в OSM не найдено присутствие ВТБ;
- зона не выглядит как аномалия высокого оборота при слишком малой клиентской базе.

Зоны с найденным ВТБ-присутствием не смешиваются с новыми точками. Они попадают в `v2_shortlist_capacity_existing_network.csv` как сценарий усиления существующей сети.

## Карта

Карта `output/maps/v2_atm_ml_potential_map.html` показывает:

- все H3-зоны с цветом по `placement_score`;
- шорт-лист новых точек отдельными маркерами;
- OSM-слои по ВТБ, конкурентам и трафиковым POI;
- поиск по H3;
- разные данные при наведении и клике.

При наведении показываются операционные данные зоны. При клике показывается ML-формула и расшифровка ключевых метрик.

## Ограничения

- Модель предсказывает наличие target-сигнала, а не прямую окупаемость банкомата.
- Target отражает текущую ATM-активность, но не учитывает стоимость аренды, безопасность, инкассацию и техническую возможность установки.
- OSM может быть неполным; для промышленного решения нужен внутренний справочник ВТБ и проверенный слой конкурентов.
- H3-зона задает район поиска, а не точный адрес установки.
