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

# Callbacks

Wywołania zwrotne to obiekty, które mogą dostosować zachowanie pętli treningowej w PyTorch [`Trainer`] (ta funkcja nie została jeszcze zaimplementowana w TensorFlow), które mogą sprawdzać stan pętli treningowej (w celu raportowania postępów, logowania na TensorBoard lub innych platformach ML) i podejmować decyzje (takie jak wczesne zatrzymanie (ang. *early stopping*)).

Wywołania zwrotne są fragmentami kodu "tylko do odczytu", poza obiektem zwracanym przez [`TrainerControl`], i nie mogą zmienić niczego w pętli treningowej. W przypadku dostosowań, które wymagają zmian w pętli treningowej, powinieneś stworzyć podklasę klasy [`Trainer`] i przeciążyć odpowiednie metody (zobacz [trainer](trainer) dla przykładów).

Domyślnie `TrainingArguments.report_to` jest ustawione na `"all"`, więc [`Trainer`] użyje następujących wywołań zwrotnych:

- [`DefaultFlowCallback`], który obsługuje domyślne zachowanie logowania, zapisywania i ewaluacji.
- [`PrinterCallback`] albo [`ProgressCallback`] do wyświetlania postępu i logów (pierwszy jest używany, jeśli dezaktywujesz tqdm przez [`TrainingArguments`], w przeciwnym razie jest to drugi).
- [`~integrations.TensorBoardCallback`] jeśli tensorboard jest dostępny (przez PyTorch >= 1.4 lub tensorboardX).
- [`~integrations.WandbCallback`] jeśli [wandb](https://www.wandb.com/) jest zainstalowany.
- [`~integrations.CometCallback`] jeśli zainstalowano [comet_ml](https://www.comet.ml/site/).
- [`~integrations.MLflowCallback`] jeśli zainstalowano [mlflow](https://www.mlflow.org/).
- [`~integrations.NeptuneCallback`] jeśli zainstalowano [neptune](https://neptune.ai/).
- [`~integrations.AzureMLCallback`] jeśli zainstalowano [azureml-sdk](https://pypi.org/project/azureml-sdk/).
- [`~integrations.CodeCarbonCallback`] jeśli zainstalowano [codecarbon](https://pypi.org/project/codecarbon/).
- [`~integrations.ClearMLCallback`] jeśli zainstalowano [clearml](https://github.com/allegroai/clearml).
- [`~integrations.DagsHubCallback`] jeśli zainstalowano [dagshub](https://dagshub.com/).
- [`~integrations.FlyteCallback`] jeśli zainstalowano [flyte](https://flyte.org/).
- [`~integrations.DVCLiveCallback`] jeśli zainstalowano [dvclive](https://dvc.org/doc/dvclive).

Jeśli pakiet jest zainstalowany, ale nie chcesz korzystać z towarzyszącej mu integracji, możesz zmienić `TrainingArguments.report_to` na listę tylko tych integracji, których chcesz użyć (np. `["azure_ml", "wandb"]`).

Główną klasą implementującą wywołania zwrotne jest [`TrainerCallback`]. Pobiera ona [`TrainingArguments`] używane do tworzenia instancji [`Trainer`], może uzyskać dostęp do wewnętrznego stanu tego trenera za pośrednictwem [`TrainerState`] i może podejmować pewne działania w pętli treningowej za pośrednictwem [`TrainerControl`].


## Available Callbacks

Oto lista dostępnych [`TrainerCallback`] w bibliotece:

[[autodoc]] integrations.CometCallback
    - setup

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
    - setup

[[autodoc]] integrations.MLflowCallback
    - setup

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

[[autodoc]] integrations.DVCLiveCallback
    - setup

## TrainerCallback

[[autodoc]] TrainerCallback

Oto przykład, jak zarejestrować niestandardowe wywołanie zwrotne w PyTorch [`Trainer`]:

```python
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
)
```

Innym sposobem na zarejestrowanie wywołania zwrotnego jest użycie `trainer.add_callback()` w następujący sposób:

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# Alternatively, we can pass an instance of the callback class
trainer.add_callback(MyCallback())
```

## TrainerState

[[autodoc]] TrainerState

## TrainerControl

[[autodoc]] TrainerControl
