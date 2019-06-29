import matplotlib.pyplot as plt;

filename = "content_loss_1l.txt"

loss = [line.rstrip('\n') for line in open(filename)]

plt.plot(loss);
plt.show();
