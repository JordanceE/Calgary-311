git oring -# Calgary 311 и историски временски податоци

Овој проект ги поврзува официјалните барања за јавни услуги во Калгари со дневни историски временски податоци. Целта е преку класични методи од податочно рударење и статистичко моделирање да се одговори на четири практични прашања: дневна snow/ice побарувачка, невообичаено висока побарувачка, затворање на барањата во рок од 7 дена и профили на градските заедници.

## Структура на проектот

```text
Calgary_311_Project.ipynb       водена анализа и објаснување на чекорите
README.md                       документација и упатство за репродукција
requirements.txt               потребни Python библиотеки
environment.yml                алтернативна conda околина
work/
  download_project_data.py     преземање и агрегирање од официјалните API извори
  analyze_project.py           чистење, моделирање, евалуација и извештаи
  data/raw/                    локално преземени податоци, се создава при извршување
  data/processed/              аналитички табели, се создава при извршување
outputs/
  analysis_results.json        структурирани резултати
  model_metrics.csv            споредливи test метрики
  figures/                     графици
  calgary_311_project_report.html
paper/                          семинарска работа
presentation/                   PowerPoint презентација
```

## Извори на податоци

- City of Calgary Open Data, Calgary 311 Service Requests: https://data.calgary.ca/Services-and-Amenities/311-Service-Requests/iahh-g8bj
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

Временските податоци се преземаат за координатите 51.0447, -114.0719 и временската зона `America/Edmonton`. Се користи ERA5 за конзистентен reanalysis запис. 311 податоците се преземаат како агрегирани табели за да не се чуваат адреси или точни координати.

## Репродукција

Потребен е Python 3.12 или понов. Во PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python work\download_project_data.py
python work\analyze_project.py
jupyter lab Calgary_311_Project.ipynb
```

На macOS или Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python work/download_project_data.py
python work/analyze_project.py
jupyter lab Calgary_311_Project.ipynb
```

Преземањето бара интернет. Анализата ги чита датотеките од `work/data/raw`, ги запишува подготвените табели во `work/data/processed` и ги обновува резултатите во `outputs`.

## Методологија

### Прашања 1 и 2

Дневните snow/ice барања се дефинирани со експлицитна листа на седум service категории. Деновите без барања се задржуваат како нули. Временските и 311 податоците се спојуваат по локален календарски датум. Моделите се тренираат на 2018-2023, 2024 служи за избор на модел и threshold, а 2025 е недопрен test период.

За дневниот број се споредуваат сезонски baseline, Poisson регресија и random forest регресија. Метриките се MAE, RMSE, R2 и mean Poisson deviance. „Висока побарувачка“ значи најмалку 480 барања, што е 95-тиот перцентил пресметан само од training периодот. Поради ретката позитивна класа, главна метрика е PR-AUC, дополнета со precision, recall, F1, balanced accuracy и ROC-AUC.

### Прашање 3

Target е дали барањето не е затворено во рок од 7 дена. `closed_date`, `updated_date` и статуси создадени по поднесувањето не се predictors. Се користат само месец, ден во неделата, канал на поднесување, service, одговорна агенција и location type. Поделбата е 2021-2023 train, 2024 validation и 2025 test. Logistic regression и decision tree даваат различен однос меѓу precision и recall.

### Прашање 4

За секоја заедница се пресметуваат удели на service и agency категории, логаритам од просечниот годишен број, варијабилност и trend. Стандардизацијата спречува карактеристиките со поголем опсег автоматски да доминираат. K-means решението се проверува со silhouette, Ward agreement, composition-only sensitivity и временски Adjusted Rand Index.

## Усогласеност со материјалите од предметот

Прегледани се приложените предавања за мерки и preprocessing, сличност, основна класификација, overfitting и model evaluation, kNN и правила, ансамбли, небалансирани класи и cost-sensitive evaluation, linear regression, cluster analysis и MLOps. Проектот ги применува нивните главни принципи: чистење и трансформација пред моделирање; одвоени train/validation/test периоди; baseline и контрола на overfitting; precision, recall, ROC-AUC и PR-AUC за ретка класа; регресија и дрва/ансамбли за надгледувано учење; стандардизација, k-means, silhouette и stability checks за кластерирање; и репродуктивна околина со дискусија за monitoring и drift.

Не е применет секој алгоритам од предавањата. kNN, rule-based classification и boosting не се додадени само за да се зголеми бројот на модели: кај Q2/Q3 веќе се споредуваат линеарен, tree и ensemble пристап со јасни различни trade-offs, додека кај временските податоци хронолошката валидација и интерпретацијата се поважни од широка случајна algorithm search. Poisson регресијата е избрана наместо обична линеарна регресија за главната count-цел, бидејќи предвидува ненегативни броеви и подобро одговара на природата на податоците.

## Спречување на data leakage

- Временската поделба ја почитува хронологијата.
- Test 2025 не учествува во избор на карактеристики, preprocessing, threshold или hyperparameters.
- Скалирањето, imputing, one-hot encoding и category collapsing се fit само на training периодот.
- За прашањето за 7 дена, полињата што настануваат по поднесувањето се користат само за конструирање на target.
- Дефиницијата за high-demand се пресметува само од training периодот.

## Главни резултати

- Poisson регресијата постигнува MAE 67.7 и R2 0.296 на 2025. Температурата од претходниот ден и сезонскиот момент се најсилни сигнали, а снегот додава корисна информација.
- Weather-only random forest за high-demand денови има PR-AUC 0.142, recall 80.0% и precision 9.3% при prevalence 2.4%. Моделот е можен широк ранен аларм, но произведува многу лажни позитиви.
- Logistic regression за барања што не се затвораат во 7 дена има PR-AUC 0.299 наспроти baseline 0.020. Видот на услугата и одговорната агенција носат најмногу сигнал.
- Избрани се 3 профили на заедници. Silhouette е 0.282, agreement со Ward е 0.732, а временската стабилност е умерена со ARI 0.411.

## Улога на LLM

LLM беше користен за организирање на истражувачкиот план, преглед на кодот, формулирање на објаснувањата и подготовка на документацијата и презентацијата. LLM не создаваше набљудувања, target вредности или model scores. Сите бројки во резултатите ги пресметуваат приложените Python скрипти од официјално преземени податоци. Изборите за временска поделба, leakage контрола, метрики и ограничувања се експлицитно документирани за да можат да се проверат и образложат.

## Ограничувања

ERA5 е reanalysis за една grid локација и не ја претставува секоја локална временска разлика. 311 барањата го мерат и однесувањето на пријавување. „Closed“ е административен статус и не мора да значи дека физичката работа е завршена. Кластерите немаат population denominator, па интензитетот не е per capita. Резултатите се асоцијативни и predictive, а не причински проценки.

## Литература

- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. https://doi.org/10.1023/A:1010933404324
- Hyndman, R. J., and Athanasopoulos, G. (2021). Forecasting Principles and Practice, 3rd ed. https://otexts.com/fpp3/
- Saito, T., and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 20, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7
- Hubert, L., and Arabie, P. (1985). Comparing Partitions. Journal of Classification, 2, 193-218. https://doi.org/10.1007/BF01908075
- Zhu, Y. et al. (2017). Structure of 311 Service Requests as a Signature of Urban Location. PLOS ONE, 12(10), e0186314. https://doi.org/10.1371/journal.pone.0186314
- Raj, R. et al. (2021). SWIFT: A Non-emergency Response Prediction System Using Sparse Gaussian Conditional Random Fields. Pervasive and Mobile Computing, 71, 101317. https://doi.org/10.1016/j.pmcj.2020.101317
