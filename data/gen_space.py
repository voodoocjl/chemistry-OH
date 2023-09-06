import pickle
import random

# with open('search_space_3', 'rb') as file:
#     search_space = pickle.load(file)
# # random.shuffle(search_space)
# print(len(search_space))

# with open('search_space_1', 'rb') as file:
#     search_space_1 = pickle.load(file)
# # random.shuffle(search_space)
# print(len(search_space_1))
# j = 0
# k = 0
# for i in search_space:    
#     if i not in search_space_1:
#         search_space_1.append(i)
#     k+=1
#     print(k)

with open('search_space', 'rb') as file:
    search_space = pickle.load(file)
# random.shuffle(search_space)

# training_data = search_space[:10000]

# with open('train_space_1', 'wb') as file:
#     pickle.dump(training_data, file)

training_data = search_space[40000:100000]

with open('data/train_space_3', 'wb') as file:
    pickle.dump(training_data, file)

# with open('search_space_shuffle', 'wb') as file:
#     pickle.dump(search_space, file)