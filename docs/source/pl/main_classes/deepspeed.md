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

# DeepSpeed

[DeepSpeed](https://github.com/microsoft/DeepSpeed), obsługiwana przez Zero Redundancy Optimizer (ZeRO), to biblioteka optymalizująca trening i dopasowywanie (ang. *fitting*) bardzo dużych modeli pracujących na GPU. Jest ona dostępna w kilku etapach ZeRO, gdzie każdy z nich stopniowo oszczędza coraz więcej pamięci GPU poprzez partycjonowanie stanu optymalizatora, gradientów oraz parametrów, co umożliwia odciążenie CPU lub NVMe. DeepSpeed jest zintegrowany z klasą [`Trainer`], a większość konfiguracji jest wykonywana automatycznie.

Jednakże, jeśli chcesz używać DeepSpeed bez [`Trainer`], Transformers posiada klasę [`HfDeepSpeedConfig`].

<Tip>

Dowiedz się więcej o korzystaniu z DeepSpeed z [`Trainer`] w przewodniku [DeepSpeed](../deepspeed).

</Tip>

## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all
