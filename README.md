

[<img src="https://media1.tenor.com/images/03437a3a7a4a9084caecf563850e3569/tenor.gif?itemid=9054712">](https://media1.tenor.com/)

## Kocasm : korean automatic sarcasm detection

- Why this name? **Kocasm** is blend word, Korean + sarcasm

### Preparing the data

- HTML data gathered from a twitter
- korean data, queries for hashtags such as **역설, 아무말, 운수좋은날, 笑, 뭐래 아닙니다, 그럴리없다, 어그로, irony sarcastic, sarcasm** yielded my corpora. And I preprocessed dataset (1) user anonymous (2) removing hashtag (3) removing url process.
<img src="https://github.com/SpellOnYou/korean-sarcasm/blob/master/img/pipeline_clean_tokens.png" width="500" height="700" alt="preprocessing-pipeline" >
If you have any other questions with corpus, please contacts me

 - ghosh: This english dataset collected by Aniruddha Ghosh and Tony Veale. See their [repository](https://github.com/AniSkywalker/SarcasmDetection) and [paper, Fracking Sarcasm using Neural Network](http://www.aclweb.org/anthology/W16-0425)
 
### Language Model (It is still being editting)

- [`bag_of_words.py`](https://github.com/SpellOnYou/korean-sarcasm/blob/master/models/bag_of_words.py): Basic bayesian model
- [`dl_models.py`](https://github.com/SpellOnYou/korean-sarcasm/blob/master/models/dl_models.py): Model classes for a general transformer
- [`tf_attention_models.py`](https://github.com/SpellOnYou/korean-sarcasm/blob/master/models/tf_attention_models.py) : Tensorflow attentive rnn model
- I'm strongly inspired by [MirunaPislar](https://github.com/MirunaPislar/Sarcasm-Detection)'s code and I referred a lot to that codes, but I tried to make my codes more pythonic and pytorchic style. Actually, I am still modifying the code.
        
* Pikasm is compatible with: Python 2.7-3.7


### Hierarchical Attention Networks

If you want your own data, you can use it.

```
export DATA_DIR=/path/to/data
export PREP_DIR=/path/to/preprocess
export SAVE_DIR=/path/to/save

python tf_attention_models.py \
    --mode train \
    --model_cfg config/attention_base.json \
    --data_file $DATA_DIR/jiwon/train.csv \
    --test_file $DATA_DIR/jiwon/test.csv \
    --pretrain_file $BERT_PRETRAIN \
    --vocab PREP_DIR/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 128
```

## See also

### linguistic, computer science related to sarcasm
   * [universal irony detection model with czech](https://pdfs.semanticscholar.org/0c27/64756299a82659605b132aef9159f61a4171.pdf)
   * [Chinese and attentive-RNN](https://link.springer.com/chapter/10.1007/978-3-319-56608-5_45)
   * [Focus on meaning conflict with hashtags](https://www.researchgate.net/publication/255729692_The_perfect_solution_for_detecting_sarcasm_in_tweets_not)
   
   Implementation as proposed by Yang et al. in "Hierarchical Attention Networks for Document Classification" (2016)
  
