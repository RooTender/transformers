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

# Processors

Procesory mogą oznaczać dwie różne rzeczy w bibliotece Transformers:
- obiekty, które wstępnie przetwarzają dane wejściowe dla modeli multimodalnych, takich jak [Wav2Vec2](../model_doc/wav2vec2) (mowa i tekst) lub [CLIP](../model_doc/clip) (tekst i wizja).
- wycofane obiekty, które były używane w starszych wersjach biblioteki do wstępnego przetwarzania danych dla GLUE lub SQUAD.

## Multi-modal processors

Każdy model multimodalny będzie wymagał obiektu do kodowania lub dekodowania danych, które grupują kilka modalności (między tekstem, wizją i dźwiękiem). Jest to obsługiwane przez obiekty zwane procesorami, które grupują dwa lub więcej obiektów przetwarzania, takich jak tokenizatory (dla modalności tekstowej), procesory obrazu (dla wizji) i ekstraktory cech (dla dźwięku).

Procesory te dziedziczą z następującej klasy bazowej, która implementuje funkcje zapisywania i ładowania:

[[autodoc]] ProcessorMixin

## Deprecated processors

Wszystkie procesory są zgodne z tą samą architekturą, którą jest  [`~data.processors.utils.DataProcessor`]. Zwraca ona listę [`~data.processors.utils.InputExample`]. Te [`~data.processors.utils.InputExample`] mogą zostać przekonwertowane na [`~data.processors.utils.InputFeatures`] w celu wprowadzenia ich do modelu.

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) to test benchmarkowy, który ocenia wydajność modeli w zróżnicowanym zestawie istniejących zadań NLU (ang. *Natural Language Understading*). Został on wydany wraz z artykułem [GLUE: A multi-task benchmark and analysis platform for natural language understanding] (https://openreview.net/pdf?id=rJ4km2R5t7).

Biblioteka ta obsługuje łącznie 10 procesorów dla następujących zadań: MRPC, MNLI, MNLI (bez dopasowania), CoLA, SST2, STSB, QQP, QNLI, RTE i WNLI.

Te procesory to:

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

Dodatkowo, poniższa metoda może być użyta do wczytania wartości z pliku danych i przekonwertowania ich na listę [`~data.processors.utils.InputExample`].

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI

[The Cross-Lingual NLI Corpus (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) jest benchmarkiem, który ocenia jakość międzyjęzykowych reprezentacji tekstu. XNLI jest zbiorem danych stworzonym przez społeczność (crowd-sourcing), opartym na [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/): pary tekstów są oznaczone tekstowymi adnotacjami dla 15 różnych języków (w tym zarówno języków dla których istnieje wiele danych treningowych, takich jak angielski i przeciwnie, takich jak suahili).

Został on opublikowany wraz z artykułem [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053).

Ta biblioteka obsługuje procesor do wczytywania danych XNLI:

- [`~data.processors.utils.XnliProcessor`]

Należy pamiętać, że ponieważ "złote" etykiety (ręcznie przygotowywane) są dostępne w zbiorze testowym to ewaluacja jest przeprowadzana również na zbiorze testowym.

Przykład użycia tych procesorów podano w skrypcie [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py).


## SQuAD

[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) to benchmark, który ocenia wydajność modeli odpowiadających na pytania (ang. *Question Answering*). Dostępne są dwie wersje, v1.1 i v2.0. Pierwsza wersja (v1.1) została wydana wraz z artykułem [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). Druga wersja (v2.0) została wydana wraz z artykułem [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822).

Ta biblioteka zawiera procesor dla każdej z dwóch wersji:

### Processors

Te procesory to:

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

Obie dziedziczą po abstrakcyjnej klasie [`~data.processors.utils.SquadProcessor`].

[[autodoc]] data.processors.squad.SquadProcessor
    - all

Dodatkowo, poniższa metoda może być użyta do konwersji przykładów SQuAD na [`~data.processors.utils.SquadFeatures`], które mogą być użyte jako dane wejściowe modelu.

[[autodoc]] data.processors.squad.squad_convert_examples_to_features


Procesory te, jak również wspomniana wcześniej metoda, mogą być używane z plikami zawierającymi dane, jak również z paczką *tensorflow_datasets*. Przykłady podano poniżej.


### Example usage

Poniżej znajduje się przykład wykorzystujący procesory, a także metodę konwersji przy użyciu plików danych:

```python
# Loading a V2 processor
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# Loading a V1 processor
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Używanie *tensorflow_datasets* jest tak proste, jak używanie pliku danych:

```python
# tensorflow_datasets only handle Squad V1.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Inny przykład użycia tych procesorów podano w skrypcie [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py).
