<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Logging

🤗 Transformers ma scentralizowany system logowania, dzięki czemu można łatwo skonfigurować ilość detali w logach (ang. verbosity) biblioteki.

Obecnie domyślną wartością ilości szczegółów logów biblioteki jest `WARNING`.

Aby zmienić poziom szczegółowości, wystarczy użyć jednego z bezpośrednich setterów. Na przykład, oto jak zmienić poziom ilości detali na poziom INFO.

```python
import transformers

transformers.logging.set_verbosity_info()
```

Można również użyć zmiennej środowiskowej `TRANSFORMERS_VERBOSITY`, aby zastąpić domyślną szczegółowość logów. Można ją ustawić na jedną z następujących wartości: `debug`, `info`, `warning`, `error`, `critical`. Na przykład:

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

Dodatkowo, niektóre `ostrzeżenia` mogą być wyłączone poprzez ustawienie zmiennej środowiskowej `TRANSFORMERS_NO_ADVISORY_WARNINGS` na wartość true, jak *1*. Spowoduje to wyłączenie każdego ostrzeżenia, które jest logowane przy użyciu [`logger.warning_advice`]. Na przykład:

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

Poniżej znajduje się przykład użycia tego samego loggera i to jak możesz go użyć w module lub skrypcie:

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```


Wszystkie metody tego modułu logowania są udokumentowane poniżej. Główne z nich to [`logging.get_verbosity`], aby uzyskać aktualny poziom szczegółowości logów i [`logging.set_verbosity`], aby ją ustawić. W kolejności (od najmniejszej szczegółowości do największej), te poziomy (z odpowiadającymi im wartościami int w nawiasach) to:

- `transformers.logging.CRITICAL` lub `transformers.logging.FATAL` (wartość int, 50): raportuje tylko błędy krytyczne.
- `transformers.logging.ERROR` (wartość int, 40): zgłasza tylko błędy.
- `transformers.logging.WARNING` lub `transformers.logging.WARN` (wartość int, 30): zgłasza tylko błędy i ostrzeżenia. Jest to domyślny poziom używany przez bibliotekę.
- `transformers.logging.INFO` (wartość int, 20): raportuje błędy, ostrzeżenia i podstawowe informacje.
- `transformers.logging.DEBUG` (wartość int, 10): raportuje wszystkie informacje.

Domyślnie, paski postępu `tqdm` będą wyświetlane podczas pobierania modelu. [`logging.disable_progress_bar`] i [`logging.enable_progress_bar`] mogą być użyte do wyłączenia lub wyłączenia tego zachowania.

## `logging` vs `warnings`

Python posiada dwa systemy logowania, które są często używane w połączeniu: `logging`, który został wyjaśniony powyżej, oraz `warnings`, który pozwala na dalszą klasyfikację ostrzeżeń w określonych zbiorach. Np. `FutureWarning` jest dla funkcji lub ścieżki, która została już wycofana, natomiast `DeprecationWarning` informuje, że jest planowane wycofanie jakiejś funkcjonalności.

Używamy obu w bibliotece `transformers`. Wykorzystujemy i dostosowujemy metodę `captureWarning` biblioteki `logging`, aby umożliwić zarządzanie tymi komunikatami ostrzegawczymi za pomocą powyższych setterów szczegółowości logów.

Co to oznacza dla twórców biblioteki? Powinniśmy przestrzegać następującej heurystyki:
- `warnings` powinny być preferowane dla deweloperów tej biblioteki i bibliotek zależnych od `transformers`
- `logging` powinno być używane dla użytkowników końcowych biblioteki, używających jej we własnych projektach

Zobacz opis metody `captureWarnings` poniżej.

[[autodoc]] logging.captureWarnings

## Base setters

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## Other functions

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar
