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

# Trainer

Klasa [`Trainer`] oferuje kompleksowe API do trenowania modeli w PyTorch, wspierając jednocześnie trening rozproszony na wielu GPU/TPU, mieszaną precyzję dla [NVIDIA GPUs](https://nvidia.github.io/apex/), [AMD GPUs](https://rocm.docs.amd.com/en/latest/rocm.html) i [`torch.amp`](https://pytorch.org/docs/stable/amp.html) dla PyTorch. Klasa [`Trainer`] działa w parze z klasą [`TrainingArguments`], która oferuje szeroki zakres opcji dostosowywania sposobu trenowania modelu. Razem te dwie klasy zapewniają kompletne API trenowania.

[`Seq2SeqTrainer`] i [`Seq2SeqTrainingArguments`] dziedziczą funkcjonalności z klas [`Trainer`] i [`TrainingArgument`] oraz są przystosowane do trenowania modeli dla zadań sekwencja-do-sekwencji (ang. *sequence-to-sequence*), takich jak streszczanie czy tłumaczenie.

<Tip warning={true}>

Klasa [`Trainer`] jest zoptymalizowana dla modeli 🤗 Transformers i może mieć niespodziewane zachowania gdy jest używana z innymi modelami. Używając jej z własnym modelem, należy się upewnić:

- Twój model zawsze zwraca krotki lub podklasy [`~utils.ModelOutput`]
- Twój model może obliczyć funkcję strat, jeśli podany zostanie argument `labels` i strata ta zostanie zwrócona jako pierwszy element krotki (jeśli model zwraca krotki).
- Twój model może przyjąć wiele argumentów w formie etykiet (użyj `label_names` w [`TrainingArguments`], aby określić ich nazwę dla instancji [`Trainer`]), ale żaden z nich nie powinien być nazwany `"label"`.

</Tip>

## Trainer[[api-reference]]

[[autodoc]] Trainer
    - all

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all
