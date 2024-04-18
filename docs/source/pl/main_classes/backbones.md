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

# Backbone

Szkielet (ang. *Backbone*) jest modelem używanym do ekstrakcji cech dla zadań wizji komputerowej wyższego poziomu, takich jak wykrywanie obiektów i klasyfikacja obrazów. Transformers udostępnia klasę [`AutoBackbone`] do inicjalizacji modelu z Transformers na podstawie wstępnie wytrenowanych wag modelu oraz dwie klasy użytkowe:

* [`~utils.BackboneMixin`] umożliwia inicjalizację szkieletu z Transformers lub [timm](https://hf.co/docs/timm/index) i zawiera funkcje do zwracania wyjściowych cech i indeksów.
* [`~utils.BackboneConfigMixin`] ustawia wyjściowe cechy i indeksy konfiguracji szkieletu.

Modele [timm](https://hf.co/docs/timm/index) są wczytywane z klasami [`TimmBackbone`] i [`TimmBackboneConfig`].

Backbones są obsługiwane dla następujących modeli:

* [BEiT](..model_doc/beit)
* [BiT](../model_doc/bit)
* [ConvNet](../model_doc/convnext)
* [ConvNextV2](../model_doc/convnextv2)
* [DiNAT](..model_doc/dinat)
* [DINOV2](../model_doc/dinov2)
* [FocalNet](../model_doc/focalnet)
* [MaskFormer](../model_doc/maskformer)
* [NAT](../model_doc/nat)
* [ResNet](../model_doc/resnet)
* [Swin Transformer](../model_doc/swin)
* [Swin Transformer v2](../model_doc/swinv2)
* [ViTDet](../model_doc/vitdet)

## AutoBackbone

[[autodoc]] AutoBackbone

## BackboneMixin

[[autodoc]] utils.BackboneMixin

## BackboneConfigMixin

[[autodoc]] utils.BackboneConfigMixin

## TimmBackbone

[[autodoc]] models.timm_backbone.TimmBackbone

## TimmBackboneConfig

[[autodoc]] models.timm_backbone.TimmBackboneConfig
