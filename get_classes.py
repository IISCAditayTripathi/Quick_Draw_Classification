import pickle


draw_classes = pickle.load(open('label2key_draw.pkl', 'rb'))

imagenet_classes = open('imagenet_classes.txt').readlines()


a = list(draw_classes.keys())
print(a)
total_classes = []
#for key, classes in imagenet_classes.items():
#    total_classes.extend(classes)
for line in imagenet_classes:
    b = line.split(':')
    b = b[1].split(',')
    b = b[0:-1]
    b[0] = b[0].strip()[1:]
    b[-1] = b[-1].strip()[:-1]
    b1 = [b2.strip().lower() for b2 in b]
    total_classes.extend(b1)
print(total_classes)
print(set(a).difference(set(total_classes)))
print(len(set(a).difference(set(total_classes))))
