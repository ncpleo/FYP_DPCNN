# Deep Pyramid Convolutional Neural Networks for Text Categorization

> This is a simple version of the paper *Deep Pyramid Convolutional Neural Networks for Text Categorization*.


!['model'](./pictures/figure1.png)

Please download the GloVe "glove.6B.zip" first, from https://nlp.stanford.edu/projects/glove/, then extract and place "glove.6B.300d.txt" into /data/glove/

run by

```
python main.py --lr=0.001 --epoch=20 --batch_size=64 --gpu=0 --seed=0 --label_num=2
```