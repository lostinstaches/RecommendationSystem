import sys

in_ = open('de_data_train.csv', 'r')
out = open('train_surprise.csv', 'w')

for row in in_.readlines()[1:]:
    
    tmp = row.split(',')
    user_and_item = tmp[0]
    user = user_and_item.split('_')[1][1:]
    item = user_and_item.split('_')[0][1:]
    rating = tmp[1]
    out.write('{},{},{}'.format(user, item, rating))

in_.close()
out.close()


