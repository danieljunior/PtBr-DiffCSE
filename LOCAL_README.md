To run the training of DiffCSE it's necessary to create another virtual env and follow the instructions from the official repository:
   - conda create -n diffcse python=3.9
   - conda activate diffcse

### Training
- Original train script
     ```    
       python train.py \
                     --model_name_or_path bert-base-uncased \
                     --generator_name distilbert-base-uncased \
                     --train_file data/wiki1m_for_simcse.txt \
                     --output_dir ./models/en_diff_cse \
                     --num_train_epochs 2 \
                     --per_device_train_batch_size 64 \
                     --learning_rate 7e-6 \
                     --max_seq_length 32 \
                     --evaluation_strategy steps \
                     --metric_for_best_model stsb_spearman \
                     --load_best_model_at_end \
                     --eval_steps 125 \
                     --pooler_type cls \
                     --mlp_only_train \
                     --overwrite_output_dir \
                     --logging_first_step \
                     --logging_dir log \
                     --temp 0.05 \
                     --do_train \
                     --do_eval \
                     --batchnorm \
                     --lambda_weight 0.005 \
                     --fp16 \
                     --masking_ratio 0.30
     ```
- Portuguese trainning script
     ``` 
       python train.py \
                 --model_name_or_path neuralmind/bert-base-portuguese-cased \
                 --generator_name adalbertojunior/distilbert-portuguese-cased \
                 --train_file data/train/pt-br.txt \
                 --output_dir ./models/portuguese_diff_cse \
                 --num_train_epochs 2 \
                 --per_device_train_batch_size 64 \
                 --learning_rate 7e-6 \
                 --max_seq_length 32 \
                 --evaluation_strategy steps \
                 --metric_for_best_model stsb_spearman \
                 --load_best_model_at_end \
                 --eval_steps 125 \
                 --pooler_type cls \
                 --mlp_only_train \
                 --overwrite_output_dir \
                 --logging_first_step \
                 --logging_dir log \
                 --temp 0.05 \
                 --do_train \
                 --do_eval \
                 --batchnorm \
                 --lambda_weight 0.005 \
                 --fp16 \
                 --masking_ratio 0.30
     ```

### Usage

#### Custom API

```
from diffcse import DiffCSE
model = DiffCSE('./models/portuguese_diff_cse')
embeddings = model.encode("A woman is reading.")
sentences_a = ['Um tigre está andando ansiosamente em torno de uma gaiola',
               'Um jovem alpinista está fazendo uma parede de escalada para os meninos']
sentences_b =  ['Um grupo de meninos em um quintal está brincando e um homem está de pé ao fundo',
                'Os meninos jovens estão brincando ao ar livre e o homem está sorrindo por perto']
similarities = model.similarity(sentences_a, sentences_b)
print(similarities)
similarities = model.similarity('Um grupo de meninos em um quintal está brincando e um homem está de pé ao fundo', \
                                 'Os meninos jovens estão brincando ao ar livre e o homem está sorrindo por perto')
print(similarities)

```
_sources:_ 
- https://github.com/voidism/DiffCSE?tab=readme-ov-file#pretrained-models
- https://github.com/princeton-nlp/SimCSE?tab=readme-ov-file#getting-started

#### HuggingFace API

```
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("./models/portuguese_diff_cse")
model = AutoModel.from_pretrained("./models/portuguese_diff_cse")

# Tokenize input texts
texts = ['Um tigre está andando ansiosamente em torno de uma gaiola',
          'Um jovem alpinista está fazendo uma parede de escalada para os meninos',
          'Um grupo de meninos em um quintal está brincando e um homem está de pé ao fundo',
           'Os meninos jovens estão brincando ao ar livre e o homem está sorrindo por perto']
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```
_source: https://github.com/princeton-nlp/SimCSE?tab=readme-ov-file#use-simcse-with-huggingface_