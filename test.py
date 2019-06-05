
import torch
import torch.utils.data
from nn_1 import add_to_data


normal_data = open("normal.txt", "r")
tumor_data = open("tumor.txt", "r")


training_data = []
add_to_data(training_data, 6, normal_data, tumor_data)

#training_data = [('Matt', 20), ('Karim', 30), ('Maya', 40)]
train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=2)


def show_batch():
	for step, (batch_x, batch_y) in enumerate(train_loader):  # for each training step
		print(list(list(zip(*batch_x))[1]))

def show_batch2():
	for epoch in range(3): 
		for step, (batch_x, batch_y) in enumerate(training_data):  # for each training step
			# train your data...
			print('Epoch:', epoch, '| Step: ', step, '| batch x: ', len(batch_x), '| batch y: ', batch_y)
show_batch()
print("-" * 10)
show_batch2()

normal_data.close()
tumor_data.close()
