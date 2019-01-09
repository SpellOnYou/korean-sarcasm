### Pikasm : korean sarcasm detector on twitter : And i love pikachu

[<img src="https://media1.tenor.com/images/03437a3a7a4a9084caecf563850e3569/tenor.gif?itemid=9054712">](https://media1.tenor.com/)

* Why this name? **Pikasm** is blend word, pikachu + sarcasm

## Overview
 - ghosh: This english dataset collected by Aniruddha Ghosh and Tony Veale. See their [repository](https://github.com/AniSkywalker/SarcasmDetection) and [paper, Fracking Sarcasm using Neural Network](http://www.aclweb.org/anthology/W16-0425)

- jiwon : This is korean data. Queries for hashtags such as **역설, 아무말, 운수좋은날, 笑, 뭐래 아닙니다, 그럴리없다, 어그로, irony sarcastic, sarcasm** yielded my corpora. And I preprocessed dataset (1) user anonymous (2) removing hashtag (3) removing url process.
    
    ![image](/images/pipeline_clean_tokens.png)
    
If you have any other questions with corpus, plz email me
        
* Pikasm is compatible with: Python 2.7-3.6.

## Overview

This contains 9 python files. (It is still being added)
- [`bag_of_words.py`](./bag_of_words.py) : 
- [`classifiers.py`](./classifiers.py) : 
- [`dl_models.py`](./dl_models.py) : Model classes for a general transformer
- [`tf_attention_models.py`](./tf_attention_models.py) : Implementation as proposed by Yang et al. in "Hierarchical Attention Networks for Document Classification" (2016)
- [`preprocess`](./preprocess.py) : 
- [`utils.py`](./utils.py) : Several utility functions
## Example Usage

### Hierarchical Attention Networks

If you want your data, you can use it.

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
    
## See also
- linguistic, computer science related to sarcasm
   * [universal irony detection model with czech](https://pdfs.semanticscholar.org/0c27/64756299a82659605b132aef9159f61a4171.pdf)
   * [Chinese and attentive-RNN](https://link.springer.com/chapter/10.1007/978-3-319-56608-5_45)
   * [Focus on meaning conflict with hashtags](https://www.researchgate.net/publication/255729692_The_perfect_solution_for_detecting_sarcasm_in_tweets_not)
Francesco Barbieri, Francesco Ronzano, and Horacio Saggion. 2014. Italian irony detection in twitter: a first approach. In The First Italian Conference on Computational Linguistics CLiC-it 2014 & the Fourth International Workshop EVALITA. 28–32.
Peng Liu, Wei Chen, Gaoyan Ou, Tengjiao Wang, Dongqing Yang, and Kai Lei. 2014. Sarcasm Detection in Social Media Based on Imbalanced Classification. In Web-Age Information Management. Springer, 459–471.
Shin, Hyopil, Munhyong Kim, Yu-Mi Jo, Hayeon Jang, and Andrew Cattle. 2013. KOSAC(Korean Sentiment Analysis Corpus): 한국어 감정 및 의견 분석 코퍼스, Information and Compuation, pages 181-190.
Nikita Desai and Anandkumar D Dave. 2016. Sarcasm Detection in Hindi sentences using Support Vector machine. International Journal 4, 7.
Rachel Giora. 1995. On irony and negation. Discourse Processes, 19(2):239–264.
Cynthia Van Hee, Els Lefever, and Veronique Hoste. 2016b. Monday mornings are my fave :) #not ex- ploring the automatic recognition of irony in en- glish tweets. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 2730–2739, Osaka, Japan. The COLING 2016 Organizing Committee.
Tony Veale and Yanfen Hao. 2010. Detecting Ironic Intent in Creative Comparisons.. In ECAI, Vol. 215. 765–770.
CC Liebrecht, FA Kunneman, and APJ van den Bosch. 2013. The perfect solution for detecting sarcasm in tweets# not.
Ashequl Qadir and Ellen Riloff. 2014. Learning emotion indicators from tweets: Hashtags, hash- tag patterns, and phrases. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1203–1209.
