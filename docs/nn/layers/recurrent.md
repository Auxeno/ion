# Recurrent

Recurrent layers over `(batch, time, features)` inputs. `RNN`, `LSTM`, and `GRU` scan a whole sequence and return outputs plus the final state; the matching `*Cell` classes apply a single timestep and expose an `initial_state` property. Pass `hx` to supply a custom initial state; see [Reference](../index.md#recurrent-state).

::: ion.nn.RNN

::: ion.nn.LSTM

::: ion.nn.GRU

::: ion.nn.RNNCell

::: ion.nn.LSTMCell

::: ion.nn.GRUCell
