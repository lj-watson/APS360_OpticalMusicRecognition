import matplotlib.pyplot as plt

# Link to best model checkpoint (~600MB)
# https://drive.google.com/file/d/1T9W8YxLXqoEjPBG-OlK3I2AzZ6LmEII0/view?usp=sharing

# Data
epochs = range(1, 16)
train_accuracy = [
    0.97596905604477, 0.9941157106410995, 0.9951032836803555, 0.9983540449345732,
    0.9990535758373796, 0.9996708089869146, 0.999629660110279, 0.9997942556168217,
    0.999876553370093, 0.9999177022467287, 0.9996708089869146, 0.9993416179738294,
    0.9997942556168217, 0.9998354044934573, 0.9998354044934573
]
validation_accuracy = [
    0.945635528330781, 0.9601837672281777, 0.9609494640122511, 0.9686064318529862,
    0.9686064318529862, 0.9655436447166922, 0.9716692189892803, 0.9732006125574273,
    0.9716692189892803, 0.9678407350689127, 0.9732006125574273, 0.9663093415007658,
    0.9732006125574273, 0.9709035222052067, 0.9762633996937213
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Train', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.savefig("tve")