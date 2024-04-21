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

# Pipelines

Potoki to świetny i łatwy sposób na wykorzystanie modeli do wnioskowania. Są to obiekty, które abstrahują większość złożonego kodu z biblioteki, oferując prosty interfejs API dedykowany kilku zadaniom, w tym rozpoznawaniu nazwanych przedmiotów (ang. *Named Entity Recognition*), przewidywaniu zamaskowanych słów (ang. *Masked Language Modeling*), analizie nastrojów (ang. *Sentiment Analysis*), ekstrakcji cech i odpowiadaniu na pytania. Przykłady użycia można znaleźć w rozdziale dot. [streszczania](../task_summary).

Istnieją dwie kategorie abstrakcji potoków, o których należy pamiętać:

- Potok [`pipeline`], który jest najpotężniejszym obiektem hermetyzującym wszystkie inne potoki.
- Potoki specyficzne dla zadań są dostępne dla zadań [audio](#audio), [wizji komputerowej](#computer-vision), [przetwarzania języka naturalnego](#natural-language-processing) i [multimodalności](#multimodal).

## The pipeline abstraction

Abstrakcja *pipeline* jest wrapperem dla innych dostępnych potoków. Jest on instancjonowany jak każdy inny, ale może to nieco ułatwić życie.

Proste wywołanie jednego działania:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

Jeśli chcesz użyć określonego modelu z [hubu](https://huggingface.co) to możesz zignorować określanie zadania, jeśli taki model posiada je domyślnie:

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

Aby wywołać potok na wielu elementach, można wywołać go za pomocą *list*.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

Aby iterować po pełnych zbiorach danych, zaleca się bezpośrednie użycie `dataset`. Oznacza to, że nie musisz alokować całego zbioru danych na raz, ani nie musisz samodzielnie wykonywać batchingu. Powinno to działać tak samo szybko jak niestandardowe cykle na GPU. Jeśli tak nie jest, nie wahaj się utworzyć zgłoszenia.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

Dla ułatwienia obsługi możliwe jest również użycie generatora:


```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # This could come from a dataset, a database, a queue or HTTP request
        # in a server
        # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
        # to use multiple threads to preprocess data. You can still have 1 thread that
        # does the preprocessing while the main runs the big inference
        yield "This is a test"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

[[autodoc]] pipeline

## Pipeline batching

Wszystkie potoki mogą używać batchingu. Będzie to działać za każdym razem, gdy potok użyje strumieniowania (a więc podczas przekazywania list lub `Dataset` lub `generator`).

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # Exactly the same output as before, but the content are passed
    # as batches to the model
```

<Tip warning={true}>

Jednakże, nie oznacza to automatycznego wzrostu wydajności. Może to być 10-krotne przyspieszenie lub 5-krotne spowolnienie w zależności od sprzętu, danych i aktualnie używanego modelu.

Przykład, gdzie głównie następuje przyspieszenie:

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)
```

Przykład, gdzie głównie następuje spowolnienie:

```python
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

Niekiedy następują bardzo długie zdania do analizy. W takim przypadku **cały** wsad będzie musiał mieć długość 400 tokenów, zatem wsad będzie miał wymiary [64, 400] zamiast [64, 4], co prowadzi do sporego spowolnienia. Co gorsza, przy większych wsadach program po prostu się zawiesza.


```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

Nie ma dobrych (w ogólności) rozwiązań dla tego problemu, a przebieg może się różnić w zależności od przypadków użycia.

Dla użytkowników główną zasadą jest:

- **Mierz wydajność na swoim obciążeniu, na swoim sprzęcie. Mierz, mierz i nie przestawaj mierzyć. Tylko liczby powiedzą Ci prawdę.**
- Jeśli masz opóźnienia (np. produkt wnioskowuje na żywo), nie stosuj wsadów.
- Jeśli korzystasz z CPU, nie stosuj wsadów.
- Jeśli używasz przepustowości (chcesz uruchomić swój model na wielu statycznych danych), na GPU, to:

  - Jeśli nie masz pojęcia o rozmiarze `sequence_length` ("naturalnych" danych), domyślnie nie stosuj wsadów. Mierz i eksperymentuj z coraz większym wsadem. Dodaj kontrole OOM (ang. *Out Of Memory*), aby odzyskać progres, gdy coś się nie powiedzie (i tak się stanie w pewnym momencie, jeśli nie kontrolujesz `sequence_length`).
  - Jeśli twoja `sequence_length` jest super regularna, to użycie wsadów prawdopodobne będzie bardzo korzystne. Mierz i zwiększaj ich rozmiar, aż uzyskasz OOM.
  - Im masz większe GPU, tym większe partie danych można przetwarzać. Warto eksperymentować z większymi wsadami.
- Jak stosujesz batching to upewnij się, że dobrze obsługujesz OOM.

## Pipeline chunk batching

`zero-shot-classification` i `question-answering` są nieco specyficzne w tym sensie, że pojedyncze dane wejściowe mogą dawać wiele przejść modelu do przodu (ang. *forward pass*). W normalnych okolicznościach spowodowałoby to problemy z argumentem `batch_size`.

Aby obejść ten problem użyj `ChunkPipeline` zamiast zwykłego `Pipeline`. W skrócie:


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

Teraz staje się:


```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

Powinno to być bardzo przejrzyste w czytaniu kodu, ponieważ potoki są używane w ten sam sposób.

Jest to uproszczony widok, ponieważ potok może automatycznie obsługiwać partię! Oznacza to, że nie musisz dbać o to, ile przejść do przodu faktycznie wyzwolą dane wejściowe i możesz zoptymalizować `batch_size` niezależnie od danych wejściowych. Zastrzeżenia z poprzedniej sekcji nadal mają zastosowanie.

## Pipeline custom code

Jeśli chcesz zastąpić konkretny potok.

Nie wahaj się utworzyć zgłoszenia dla swojego zadania, celem potoków jest bycie łatwym w użyciu i wspieranie większości działań, więc `transformers` mogą wspierać również twój przypadek użycia.


Jeśli chcesz spróbować, po prostu możesz:

- Stworzyć podklasę wybranego potoku

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

Powinno to umożliwić dodanie własnego kodu.


## Implementing a pipeline

[Implementacja nowego potoku](../add_new_pipeline)

## Audio

Potoki dostępne dla zadań audio obejmują następujące elementy.

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### TextToAudioPipeline

[[autodoc]] TextToAudioPipeline
    - __call__
    - all


### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## Computer vision

Potoki dostępne dla zadań wizji komputerowej obejmują następujące elementy.

### DepthEstimationPipeline
[[autodoc]] DepthEstimationPipeline
    - __call__
    - all

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline
    - __call__
    - all

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline
    - __call__
    - all

### ImageToImagePipeline

[[autodoc]] ImageToImagePipeline
    - __call__
    - all

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline
    - __call__
    - all

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline
    - __call__
    - all

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline
    - __call__
    - all

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline
    - __call__
    - all

## Natural Language Processing

Potoki dostępne dla zadań przetwarzania języka naturalnego obejmują następujące elementy.

### ConversationalPipeline

[[autodoc]] Conversation

[[autodoc]] ConversationalPipeline
    - __call__
    - all

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline
    - __call__
    - all

### SummarizationPipeline

[[autodoc]] SummarizationPipeline
    - __call__
    - all

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline
    - __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline
    - __call__
    - all

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline
    - __call__
    - all

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline
    - __call__
    - all

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline
    - __call__
    - all

### TranslationPipeline

[[autodoc]] TranslationPipeline
    - __call__
    - all

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline
    - __call__
    - all

## Multimodal

Potoki dostępne dla zadań multimodalnych obejmują następujące elementy.

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageFeatureExtractionPipeline

[[autodoc]] ImageFeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### MaskGenerationPipeline

[[autodoc]] MaskGenerationPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## Parent class: `Pipeline`

[[autodoc]] Pipeline
