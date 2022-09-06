python3 train_atten.py \
	--device=cuda \
	--seed=42 \
	--batch_size=32 \
	--epochs=3 \
	--lr=0.1 \
	--evaluation_epochs=1 \
	--optim=Adam \
	--loss_fun=CrossEntropyLoss \
	--bert_path=/root/autodl-tmp/bert-base-uncased \
	--input_ids_filename=./hot_data/input_ids.pkl \
	--train_sentences_path=./hot_data/train_add.pkl \
	--test_sentence_path=./hot_data/test_add.pkl \
	--labels_path=./hot_data/data_set.pkl \
	--alpha=0.5 \
	--hidden_size=76