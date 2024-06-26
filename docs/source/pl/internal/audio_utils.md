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

# Narzędzia do `FeatureExtractors`

Ta strona zawiera listę wszystkich funkcji użytkowych, które mogą być używane przez audio [`FeatureExtractor`] w celu obliczenia specjalnych cech z surowego dźwięku przy użyciu popularnych algorytmów, takich jak *Krótkoczasowa transformata Fouriera* lub *spektrogram w logarytmicznej skali melowej*.

Większość z nich jest przydatna tylko wtedy, gdy studiujesz kod procesorów audio w bibliotece.

## Audio Transformations

[[autodoc]] audio_utils.hertz_to_mel

[[autodoc]] audio_utils.mel_to_hertz

[[autodoc]] audio_utils.mel_filter_bank

[[autodoc]] audio_utils.optimal_fft_length

[[autodoc]] audio_utils.window_function

[[autodoc]] audio_utils.spectrogram

[[autodoc]] audio_utils.power_to_db

[[autodoc]] audio_utils.amplitude_to_db
