# Training the team network

`../../team_tcn.npz` is a temporal convolutional network for P(home | ball alive) at 5 Hz. It reads
the same 530 features as the trees in `../../team_model.txt`, over about 12 s either side of each
row. `../../run.py` averages the two models' probabilities in log-odds. Like the trees, it is
trained only on the seven open DFL matches from `../bench/`, degraded the same way.

## Train it

From the repository root, after `../download.py` and with the bench's feature tables built (the
first run of `../bench/export_team.py` builds them):

```bash
pip install -r solution/training/network/requirements.txt          # PyTorch, for training only
python solution/training/network/train_final.py cache/team_model/team_tcn.npz 0
```

The architecture is a 1x1 convolution to 64 channels, then five residual blocks of two dilated
convolutions of width 3 (dilations 1, 2, 4, 8, 16), then a 1x1 output. Features are standardised
with the training means and standard deviations, which are stored in the `.npz`; missing values
become 0. Training uses crops of 256 rows (51 s), batches of 32 and 2,000 steps of AdamW
(learning rate 2e-3 with one-cycle schedule, weight decay 1e-4, dropout 0.1). The loss is binary
cross-entropy on the rows where the ball is alive. The training data is two degraded copies of
every match (seeds 100 and 101), each also home/away-swapped and mirrored.

`tcn_numpy.py` exports the model as plain arrays. `../../tcn.py` applies it in NumPy (float64),
which matches PyTorch to within 2e-7 in log-odds.

## The bench

Team step only, the true dead ball and our decoder, each match held out in turn:

```bash
python solution/training/network/lgb_lean.py base 100 101    # the trees, out-of-fold
python solution/training/network/tcn_train.py tcn2 100 101   # the network, out-of-fold
python solution/training/network/nested.py                   # switch cost chosen on the other six
```

| team step | mean spellF1 | better than the trees |
|---|---|---|
| trees (the shipped recipe), switch cost 1.0 | 0.586 | - |
| network, switch cost 1.0 | 0.608 | 6 of 7 |
| trees and network averaged, switch cost chosen on the other six (2.0 every time) | **0.634** | **7 of 7** |

Tried and dropped: four degraded copies per match for the trees (better on only 1 of the first
3 matches, stopped). Probabilities and results go to `cache/team_model/network/` (git-ignored).
`common.py` loads one cached feature table at a time, so the bench fits in about 6 GB of memory.
