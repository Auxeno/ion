# Recurrent

Recurrent layers over `(batch, time, features)` inputs. `RNN`, `LSTM`, and `GRU` scan a whole sequence and return outputs plus the final state; the matching `*Cell` classes apply a single timestep.

::: ion.nn.RNN

::: ion.nn.LSTM

::: ion.nn.GRU

::: ion.nn.RNNCell

::: ion.nn.LSTMCell

::: ion.nn.GRUCell

---

## Recurrent State

Sequence layers default to a zero initial state. Pass `hx` to start from a
custom state or continue a recurrence across sequence chunks.

```python
rnn = nn.RNN(3, 16, key=key)
outputs, h = rnn(x)
outputs, h = rnn(x_next, hx=h)

lstm = nn.LSTM(3, 16, key=key)
outputs, (h, c) = lstm(x)
outputs, (h, c) = lstm(x_next, hx=(h, c))
```

Cells expose an unbatched `initial_state` property:

```python
cell = nn.LSTMCell(3, 16, key=key)
h, c = cell.initial_state
```
