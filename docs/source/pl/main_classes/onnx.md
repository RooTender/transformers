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

# Exporting 🤗 Transformers models to ONNX

Transformers udostępnia pakiet `transformers.onnx`, który umożliwia konwersję punktów kontrolnych modelu na wykres ONNX poprzez wykorzystanie obiektów konfiguracyjnych.

Więcej szczegółów można znaleźć w [przewodniku](../serialization) na temat eksportowania modeli 🤗 Transformers.

## ONNX Configurations

Zapewniamy trzy klasy abstrakcyjne, z których należy dziedziczyć, w zależności od typu architektury modelu, który chcesz wyeksportować:

* Modele oparte na koderze dziedziczą po [`~onnx.config.OnnxConfig`]
* Modele oparte na dekoderach dziedziczą po [`~onnx.config.OnnxConfigWithPast`]
* Modele kodera-dekodera dziedziczą z [`~onnx.config.OnnxSeq2SeqConfigWithPast`].

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX Features

Każda konfiguracja ONNX jest powiązana z zestawem _funkcji_, które umożliwiają eksportowanie modeli dla różnych typów topologii lub zadań.

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager

