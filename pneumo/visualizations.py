import matplotlib.pyplot as plt
import tensorflow as tf


def display(display_list, titles=None, cmaps=None):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        if not titles is None:
            plt.title(titles[i])
        cmap = None
        if not cmaps is None:
            cmap = cmaps[i]
        plt.imshow(
            tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap=cmap
        )
        plt.axis("off")
    plt.show()
