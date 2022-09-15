python3 train_no_embw.py /
	--device=cuda /
	--seed=42 /
	--batch_size=64 /
	--input_size=300 /
	--hidden_size1=300 /
	--hidden_size2=300 /
	--epochs=5 /
	--lr=0.001 /
	--evaluation_epochs=1 /
	--optim=Adam /
	--loss_fun=CrossEntropyLoss /
	--notes=Initial test /
	--load_para=init.pth /
	--if_load=False /
	--if_save=False /
	--save_name=init.pth /
	--debug=Fals