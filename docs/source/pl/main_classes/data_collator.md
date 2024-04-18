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

# Data Collator

Kolatory danych to obiekty, które tworzą wsad (ang. *batch*) przy użyciu listy elementów zbioru danych jako danych wejściowych. Elementy te są tego samego typu co elementy `train_dataset` lub `eval_dataset`.

Aby móc tworzyć wsady, kolatory danych mogą stosować pewne przetwarzanie (np. wypełnianie). Niektóre z nich (jak np. [`DataCollatorForLanguageModeling`]) stosują również pewne losowe rozszerzenie danych (np. losowe maskowanie) na utworzonej partii.

Przykłady użycia można znaleźć w [example scripts](../examples) lub [example notes](../notebooks).


## Default data collator

[[autodoc]] data.data_collator.default_data_collator

## DefaultDataCollator

[[autodoc]] data.data_collator.DefaultDataCollator

## DataCollatorWithPadding

[[autodoc]] data.data_collator.DataCollatorWithPadding

## DataCollatorForTokenClassification

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## DataCollatorForSeq2Seq

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## DataCollatorForLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForWholeWordMask

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForPermutationLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens
