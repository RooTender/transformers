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

# Narzędzia do generacji

Ta strona zawiera listę wszystkich funkcji pomocnicznych używanych przez [`~generation.GenerationMixin.generate`].

## Generowanie danych wyjściowych

Wyjście [`~generation.GenerationMixin.generate`] jest instancją podklasy [`~utils.ModelOutput`].
[`~utils.ModelOutput`]. To wyjście jest strukturą danych zawierającą wszystkie informacje zwrócone przez
przez [`~generation.GenerationMixin.generate`], ale może być również użyta jako krotka lub słownik.

Przykład:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

Obiekt `generation_output` jest [`~generation.GenerateDecoderOnlyOutput`], jak możemy zobaczyć w dokumentacji tej klasy poniżej. Oznacza to, że posiada następujące atrybuty:

- `sequences`: wygenerowane sekwencje tokenów
- `scores` (opcjonalnie): wyniki predykcji głowicy modelującej język, dla każdego kroku generacji
- `hidden_states` (opcjonalnie): ukryte stany modelu, dla każdego kroku generacji
- `attentions` (opcjonalne): wagi uwagi modelu, dla każdego kroku generacji

Tutaj mamy `scores`, ponieważ przekazaliśmy `output_scores=True`, ale nie mamy `hidden_states` i `attentions`, ponieważ nie przekazaliśmy `output_hidden_states=True` lub `output_attentions=True`.

Możesz uzyskać dostęp do każdego atrybutu tak, jak zazwyczaj, a jeśli ten atrybut nie został zwrócony przez model to otrzymasz `None`. Tutaj na przykład `generation_output.scores` to wszystkie wygenerowane wyniki predykcji głowicy modelującej język, a `generation_output.attentions` to `None`.

Kiedy używamy naszego obiektu `generation_output` jako krotki, zachowuje on tylko atrybuty, które nie mają wartości `None`. Tutaj, na przykład, ma dwa elementy, `loss` i `logits`, więc

```python
generation_output[:2]
```

zwróci krotkę `(generation_output.sequences, generation_output.scores)`.

Kiedy używamy naszego obiektu `generation_output` jako słownika, przechowuje on tylko atrybuty, które nie mają wartości `None`. Tutaj, na przykład, ma dwa klucze, które są `sequences` i `scores`.

Dokumentujemy tutaj wszystkie typy wyjść.


### PyTorch

[[autodoc]] generation.GenerateDecoderOnlyOutput

[[autodoc]] generation.GenerateEncoderDecoderOutput

[[autodoc]] generation.GenerateBeamDecoderOnlyOutput

[[autodoc]] generation.GenerateBeamEncoderDecoderOutput

### TensorFlow

[[autodoc]] generation.TFGreedySearchEncoderDecoderOutput

[[autodoc]] generation.TFGreedySearchDecoderOnlyOutput

[[autodoc]] generation.TFSampleEncoderDecoderOutput

[[autodoc]] generation.TFSampleDecoderOnlyOutput

[[autodoc]] generation.TFBeamSearchEncoderDecoderOutput

[[autodoc]] generation.TFBeamSearchDecoderOnlyOutput

[[autodoc]] generation.TFBeamSampleEncoderDecoderOutput

[[autodoc]] generation.TFBeamSampleDecoderOnlyOutput

[[autodoc]] generation.TFContrastiveSearchEncoderDecoderOutput

[[autodoc]] generation.TFContrastiveSearchDecoderOnlyOutput

### FLAX

[[autodoc]] generation.FlaxSampleOutput

[[autodoc]] generation.FlaxGreedySearchOutput

[[autodoc]] generation.FlaxBeamSearchOutput

## LogitsProcessor

A [`LogitsProcessor`] can be used to modify the prediction scores of a language model head for
generation.

### PyTorch

[[autodoc]] AlternatingCodebooksLogitsProcessor
    - __call__

[[autodoc]] ClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] EncoderNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] EncoderRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] EpsilonLogitsWarper
    - __call__

[[autodoc]] EtaLogitsWarper
    - __call__

[[autodoc]] ExponentialDecayLengthPenalty
    - __call__

[[autodoc]] ForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] ForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] ForceTokensLogitsProcessor
    - __call__

[[autodoc]] HammingDiversityLogitsProcessor
    - __call__

[[autodoc]] InfNanRemoveLogitsProcessor
    - __call__

[[autodoc]] LogitNormalization
    - __call__

[[autodoc]] LogitsProcessor
    - __call__

[[autodoc]] LogitsProcessorList
    - __call__

[[autodoc]] LogitsWarper
    - __call__

[[autodoc]] MinLengthLogitsProcessor
    - __call__

[[autodoc]] MinNewTokensLengthLogitsProcessor
    - __call__

[[autodoc]] NoBadWordsLogitsProcessor
    - __call__

[[autodoc]] NoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] PrefixConstrainedLogitsProcessor
    - __call__

[[autodoc]] RepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] SequenceBiasLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TemperatureLogitsWarper
    - __call__

[[autodoc]] TopKLogitsWarper
    - __call__

[[autodoc]] TopPLogitsWarper
    - __call__

[[autodoc]] TypicalLogitsWarper
    - __call__

[[autodoc]] UnbatchedClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] WhisperTimeStampLogitsProcessor
    - __call__

### TensorFlow

[[autodoc]] TFForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForceTokensLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessorList
    - __call__

[[autodoc]] TFLogitsWarper
    - __call__

[[autodoc]] TFMinLengthLogitsProcessor
    - __call__

[[autodoc]] TFNoBadWordsLogitsProcessor
    - __call__

[[autodoc]] TFNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] TFRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TFTemperatureLogitsWarper
    - __call__

[[autodoc]] TFTopKLogitsWarper
    - __call__

[[autodoc]] TFTopPLogitsWarper
    - __call__

### FLAX

[[autodoc]] FlaxForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForceTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessorList
    - __call__

[[autodoc]] FlaxLogitsWarper
    - __call__

[[autodoc]] FlaxMinLengthLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxTemperatureLogitsWarper
    - __call__

[[autodoc]] FlaxTopKLogitsWarper
    - __call__

[[autodoc]] FlaxTopPLogitsWarper
    - __call__

[[autodoc]] FlaxWhisperTimeStampLogitsProcessor
    - __call__

## StoppingCriteria

A [`StoppingCriteria`] może być użyty do zmiany momentu zatrzymania generowania (innego niż token EOS). Należy pamiętać, że jest to dostępne wyłącznie dla naszych implementacji PyTorch.

[[autodoc]] StoppingCriteria
    - __call__

[[autodoc]] StoppingCriteriaList
    - __call__

[[autodoc]] MaxLengthCriteria
    - __call__

[[autodoc]] MaxTimeCriteria
    - __call__

## Constraints

Można użyć [`Constraint`], aby wymusić generowanie określonych tokenów lub sekwencji na wyjściu. Należy pamiętać, że jest to dostępne wyłącznie dla naszych implementacji PyTorch.

[[autodoc]] Constraint

[[autodoc]] PhrasalConstraint

[[autodoc]] DisjunctiveConstraint

[[autodoc]] ConstraintListState

## BeamSearch

[[autodoc]] BeamScorer
    - process
    - finalize

[[autodoc]] BeamSearchScorer
    - process
    - finalize

[[autodoc]] ConstrainedBeamSearchScorer
    - process
    - finalize

## Streamers

[[autodoc]] TextStreamer

[[autodoc]] TextIteratorStreamer

## Caches

[[autodoc]] Cache
    - update

[[autodoc]] DynamicCache
    - update
    - get_seq_length
    - reorder_cache
    - to_legacy_cache
    - from_legacy_cache

[[autodoc]] SinkCache
    - update
    - get_seq_length
    - reorder_cache

[[autodoc]] StaticCache
    - update
    - get_seq_length
