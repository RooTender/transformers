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

# Modele

Klasy bazowe [`PreTrainedModel`], [`TFPreTrainedModel`] i [`FlaxPreTrainedModel`] implementują wspólne metody wczytywania/zapisu modelu albo z lokalnego pliku lub katalogu, albo z konfiguracji wstępnie wytrenowanego modelu dostarczonej przez bibliotekę (pobranej z repozytorium AWS S3 HuggingFace).

[`PreTrainedModel`] i [`TFPreTrainedModel`] również implementują kilka metod, które są wspólne dla wszystkich modeli:

- zmiana rozmiaru osadzeń tokenów (ang. *token embedding*) wejściowych, gdy nowe tokeny są dodawane do słownika
- przycinanie głowic uwagi (ang. *attention heads*) modelu.

Pozostałe metody, które są wspólne dla każdego modelu są zdefiniowane w [`~modeling_utils.ModuleUtilsMixin`] (dla modeli PyTorch) i [`~modeling_tf_utils.TFModuleUtilsMixin`] (dla modeli TensorFlow) lub dla generowania tekstu, [`~generation. GenerationMixin`] (dla modeli PyTorch), [`~generation.TFGenerationMixin`] (dla modeli TensorFlow) i [`~generation.FlaxGenerationMixin`] (dla modeli Flax/JAX).


## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
