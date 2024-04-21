<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Kwantyzacja

Techniki kwantyzacji zmniejszają złożoność pamięciową i  obliczeniową, reprezentując wagi i aktywacje za pomocą typów danych o niższej precyzji, takich jak 8-bitowe liczby całkowite (int8). Umożliwia to ładowanie większych modeli, które normalnie nie byłyby w stanie zmieścić się w pamięci, i przyspiesza wnioskowanie. Transformers obsługuje algorytmy kwantyzacji AWQ i GPTQ oraz 8-bitową i 4-bitową kwantyzację za pomocą bitsandbytes.

Techniki kwantyzacji, które nie są obsługiwane w Transformers mogą być dodane za pomocą klasy [`HfQuantizer`].

<Tip>

Dowiedz się, jak kwantyzować modele w przewodniku [Kwantyzacja](../kwantyzacja).

</Tip>

## QuantoConfig

[[autodoc]] QuantoConfig

## AqlmConfig

[[autodoc]] AqlmConfig

## AwqConfig

[[autodoc]] AwqConfig

## GPTQConfig

[[autodoc]] GPTQConfig

## BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig

## HfQuantizer

[[autodoc]] quantizers.base.HfQuantizer
