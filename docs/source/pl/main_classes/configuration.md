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

# Konfiguracja

Klasa bazowa [`PretrainedConfig`] implementuje wspólne metody do wczytywania/zapisu konfiguracji z lokalnego pliku lub katalogu, lub z wstępnie wytrenowanej konfiguracji modelu dostarczonej przez bibliotekę (pobranej z repozytorium AWS S3 HuggingFace).

Każda pochodna klasa config implementuje atrybuty specyficzne dla modelu. Wspólnymi atrybutami obecnymi we wszystkich klasach config są: `hidden_size`, `num_attention_heads` i `num_hidden_layers`. Modele tekstowe dodatkowo implementują:
`vocab_size`.


## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
