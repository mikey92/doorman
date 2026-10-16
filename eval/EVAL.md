# Evaluation

Every run rates every message in `eval/messages.jsonl` with the same `triage()` the page uses. Scores are read from the model's next-token logits, so a run is reproducible. See `eval/DATA.md` for where the messages come from and how they were labelled.

## At the page's settings (pitch at log-odds ≥ 9, customer below 6)

| | Customers hidden | Customers left as unsure | Pitches folded away | Pitches left as unsure |
|---|---|---|---|---|
| gemma-3-1b-v1 | 4 of 20 | 13 | 19 of 20 | 1 |

## How well the score ranks pitches above customers

- gemma-3-1b-v1: AUC 0.910 over 40 messages

## By source

| | Customers hidden | Customers left as unsure | Pitches folded away | Pitches left as unsure |
|---|---|---|---|---|
| gemma-3-1b-v1 · forum | 4 of 20 | 13 | 19 of 20 | 1 |

## Where the line for "pitch" could go

Every message is placed by its highest-rated sentence. Moving the bar trades customers hidden against pitches caught.

### gemma-3-1b-v1

| Pitch at log-odds ≥ | Customers hidden | Pitches folded away |
|---|---|---|
| 0 | 19 of 20 | 20 of 20 |
| 2 | 18 of 20 | 20 of 20 |
| 4 | 17 of 20 | 20 of 20 |
| 6 | 17 of 20 | 20 of 20 |
| 8 | 9 of 20 | 19 of 20 |
| 9 | 4 of 20 | 19 of 20 |
| 10 | 2 of 20 | 14 of 20 |
| 12 | 0 of 20 | 9 of 20 |
