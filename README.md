# АТМосфера ВТБ

## Структура

| Путь | Назначение |
|---|---|
| `data.parquet` | Транзакционные данные. |
| `target.parquet` | Target данные. |
| `notebooks/train_ml.ipynb` | Основной ноутбук: EDA, признаки, обучение ML, скоринг, экспорт артефактов. |
| `scripts/main.py` | Те же функции и CLI-запуск для воспроизводимого пересчета. |
| `docs/v2_ml_solution.md` | Методология ML-решения. |
| `output/` | Все результаты v2: CSV, карта, графики, презентация, диагностика. |
| `requirements.txt` | Зависимости только для v2. |

## Запуск из дирректории папки

Догрузите к файлу target.parquet файл data.parquet 

```powershell
python -m pip install -r requirements.txt
python -m notebook notebooks/train_ml.ipynb
```

Повторно скачать OSM-данные:

```powershell
python scripts/main.py --refresh-osm
```

Запуск без OSM:

```powershell
python scripts/main.py --no-osm
```

## Главные артефакты

- `output/v2_solution_summary.md`
- `output/pptx/vtb_atm_geoanalytics_solution_v2_ml.pptx`
- `output/data/v2_zone_scores_ml.csv`
- `output/data/v2_cell_features_full.csv`
- `output/data/v2_shortlist_new_atm.csv`
- `output/data/v2_shortlist_capacity_existing_network.csv`
- `output/data/v2_model_feature_importance.csv`
- `output/data/v2_diagnostics.json`
- `output/maps/v2_atm_ml_potential_map.html`
