python3 train_bert.py \
	--device=cuda \
	--seed=42 \
	--batch_size=32 \
	--input_size=768 \
	--hidden_size1=768 \
	--hidden_size2=768 \
	--epochs=1 \
	--lr=0.0001 \
	--evaluation_epochs=1 \
	--optim=Adam \
	--loss_fun=CrossEntropyLoss \
	--project=DRNN-bert