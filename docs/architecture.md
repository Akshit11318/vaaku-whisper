# Architecture

Vaakku is one repository. `main` holds the code. The GPU box is started by hand. A push does not train or publish.

```mermaid
flowchart LR
  subgraph disk [Instance store /home/ubuntu/vaaku]
    Raw[FLEURS archives]
    Prepared[train validation test jsonl]
    Runs[checkpoints]
  end
  Raw --> Prepared --> Train[LoRA training]
  Test[FLEURS test jsonl] --> Eval[WER and CER]
  Train --> Eval
  Eval --> Publish[explicit publish script]
  Publish --> HF[rudra11/vaakku-whisper-small]
  HF --> API[FastAPI]
  API --> Studio[Vaakku Studio]
```

The reported test set is the official FLEURS test split. Common Voice and Kathbath, when added later, stay out of that split.

Training selects checkpoints by validation loss. Word error rate is computed by the separate evaluation script.
