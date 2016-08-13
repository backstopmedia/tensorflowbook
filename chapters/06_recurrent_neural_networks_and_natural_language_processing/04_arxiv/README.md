# arXiv character-by-character generative model

To train the model, simply run `train.py`:

```
$ python3 train.py
```

Then, to generate a sample abstract, run `sample.py`:

```
$ python3 sample.py
```

If you want to change the starting seed of the generated abstract or change its length, just modify this line in `sample.py` to your liking:

```python
print(Sampling(get_params())('<SEED>', <LENGTH>))
```

Unfortunately, at this time the sample code is only compatible with Python 3. We're working on providing Python 2 translations of the code.
