

## Install [pyenv](https://github.com/pyenv/pyenv) 

```
on Mac OS : brew install pyenv
on Linux : curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
```

Install python in pyenv

```
pyenv install 3.6.2
```

install `direnv` :https://github.com/direnv/direnv

...

#### To run char rnn 

train
```
python CharRNN/train.py \
  --use_embedding \
  --input_file CharRNN/data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```

get sample

```$xslt
python CharRNN/sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \ 
  --checkpoint_path model/poetry/ \ 
  --max_length 300
```