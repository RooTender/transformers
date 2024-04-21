<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Generowanie

Każdy framework posiada metodę generowania tekstu zaimplementowaną w odpowiedniej klasie `GenerationMixin`:

- PyTorch [`~generation.GenerationMixin.generate`] jest zaimplementowany w [`~generation.GenerationMixin`].
- TensorFlow [`~generation.TFGenerationMixin.generate`] jest zaimplementowany w [`~generation.TFGenerationMixin`].
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] jest zaimplementowany w [`~generation.FlaxGenerationMixin`].

Niezależnie od wybranego frameworka, można sparametryzować metodę generowania za pomocą instancji klasy [`~generation.GenerationConfig`]. W tej klasie znajduje się pełna lista parametrów, które kontrolują zachowanie generowania.

Aby dowiedzieć się, jak sprawdzić konfigurację generowania modelu, jakie są wartości domyślne, jak zmienić parametry ad hoc oraz jak utworzyć i zapisać niestandardową konfigurację generowania, zapoznaj się z przewodnikiem [przewodnik po strategiach generowania tekstu](../generation_strategies). Wyjaśnia on również, jak korzystać z powiązanych funkcji, takich jak strumieniowanie tokenów.

## GenerationConfig

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained
	- update
	- validate
	- get_generation_mode

## GenerationMixin

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores

## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin
	- generate
