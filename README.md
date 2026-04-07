## Мультилейбл-классификация тегов статей Arxiv по title и abstract

## 1. Данные
Данные представляют из себя ```(title, abstract, tags)```, причём сэмплируем из каждой категории тегов (всего их 8) не более 200k статей. [link](https://www.kaggle.com/datasets/Cornell-University/arxiv).

## 2. Модели

### 2.1. Embedder
*   **Модель:** `malteos/scincl` (SciNCL) — специализированная Sentence Transformer, дообученная на научных текстах.
*   **Процесс:** Сразу векторизовал все тексты, саму модель не дообучал.

### 2.2. Архитектура Классификатора
Далеe embeddings из предыдущей модели подавались в Multilabel Classification Head:
*   **Вход:** Вектор размерности 768.
*   **Архитектура:** 3-x слойный MLP c GELU
*   **Функция потерь:** `BCEWithLogitsLoss`.

### 2.3. Обучение
*   **Фреймворк:** Lightning.
*   **Оптимизатор:** AdamW с lr = 1e-3.
*   **Batch Size:** 1024.
*   **Эпохи:** 50 (лучшей оказалась модель с 11 эпохи).
*   **Метрики на валидации:**
    *   `MultilabelF1Score` (average='macro')
    *   `MultilabelHammingDistance`
    *   `MultilabelAUROC` (average='macro')
    *   `MultilabelAccuracy` (average='macro')

## 3. Результаты экспериментов

Метрики валидации:

| Метрика | Значение |
| :--- | :--- |
| **MultilabelF1Score (Macro)** | **0.765** |
| **MultilabelAUROC (Macro)** | **0.988** |
| **MultilabelHammingDistance** | **0.028** |
| **MultilabelAccuracy** | **0.971** |
